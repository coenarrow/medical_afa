import argparse
import datetime
import logging
import os
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import kits
from utils.losses import get_aff_loss
from wetr.PAR import PAR
from utils import evaluate, imutils
from utils.AverageMeter import AverageMeter
from utils.camutils import (cam_to_label, cams_to_affinity_label, ignore_img_box,
                            multi_scale_cam, multi_scale_cam_with_aff_mat,
                            propagte_aff_cam_with_bkg, refine_cams_with_bkg_v2,
                            refine_cams_with_cls_label)
from utils.optimizer import PolyWarmupAdamW
from utils.medical_utils import (get_device, setup_seed, setup_logger, cal_eta,
                                  get_down_size, get_seg_loss, get_mask_by_radius)
from wetr.model_attn_aff import WeTr

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/kits_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--pooling", default="gmp", type=str, help="pooling method")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--local_rank", default=-1, type=int, help="local_rank")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")
parser.add_argument('--backend', default='nccl')
parser.add_argument('--cpu', action='store_true', help='Force CPU device')


def generate_attention_maps_after_training(cfg, checkpoint_path, device, args):
    """Generate attention maps using the best trained checkpoint."""
    from scripts.gen_attn import generate_attention_maps, get_dataset

    logging.info(f"Loading checkpoint for attention map generation: {checkpoint_path}")

    # Create output directories
    output_dir = os.path.join(cfg.work_dir.dir, 'attention_maps')
    cam_npy_dir = os.path.join(cfg.work_dir.dir, 'cam_npy')

    # Create model (without DDP wrapper for inference)
    model = WeTr(
        backbone=cfg.backbone.config,
        stride=cfg.backbone.stride,
        num_classes=cfg.dataset.num_classes,
        embedding_dim=256,
        pretrained=False,
        pooling=args.pooling,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    dataset = get_dataset(cfg, split='val')
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False
    )

    logging.info(f"Generating attention maps for {len(dataset)} samples...")

    # Generate attention maps
    scales = list(cfg.cam.scales) if hasattr(cfg.cam, 'scales') else [1.0, 0.5, 1.5]
    generate_attention_maps(
        data_loader, model, device,
        output_dir=output_dir,
        cam_npy_dir=cam_npy_dir,
        scales=scales,
        save_attn=False,
        attn_npy_dir=None
    )

    logging.info(f"Attention maps saved to: {output_dir}")
    logging.info(f"CAM NPY files saved to: {cam_npy_dir}")

def validate(model=None, data_loader=None, cfg=None, device=None):
    """
    Validation for KITS dataset.

    NOTE: KITS has no segmentation ground truth, so only classification metrics are
    computed. Segmentation-based metrics return dummy values to match COCO signature.
    CAM/attention maps are generated only at end of training via generate_attention_maps_after_training().
    
    Returns:
        tuple: (cls_score, seg_score, cam_score, aff_score) where seg/cam/aff are dummy dicts
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader),
                            total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, labels, cls_label = data

            inputs = inputs.to(device)
            cls_label = cls_label.to(device)

            # Only compute classification output - skip CAM/seg generation during validation
            cls, segs, edge, attn_pred = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

    cls_score = avg_meter.pop('cls_score')
    
    # Return dummy scores for seg/cam/aff to match COCO signature (no GT masks available)
    dummy_score = {"pAcc": 0.0, "mAcc": 0.0, "miou": 0.0, "iou": {}}
    model.train()
    return cls_score, dummy_score, dummy_score, dummy_score


def train(cfg):

    num_workers = cfg.train.get('num_workers', 4)

    dist.init_process_group(backend=args.backend,)

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    # KITS dataset uses slice_split for index-based splitting
    train_dataset = kits.KITSClsDataset(
        root_dir=cfg.dataset.root_dir,
        split=cfg.train.split,
        stage='train',
        slice_split=cfg.dataset.slice_split,
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    # KITS validation dataset returns dummy seg labels (no ground truth available)
    val_dataset = kits.KITSClsValDataset(
        root_dir=cfg.dataset.root_dir,
        split=cfg.val.split,
        stage='val',
        slice_split=cfg.dataset.slice_split,
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset,shuffle=True)
    train_loader_kwargs = dict(
        batch_size=cfg.train.samples_per_gpu,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
        persistent_workers=True,
    )
    if num_workers > 0:
        train_loader_kwargs['prefetch_factor'] = 4
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = get_device(args.local_rank, force_cpu=args.cpu)

    wetr = WeTr(backbone=cfg.backbone.config,
                stride=cfg.backbone.stride,
                num_classes=cfg.dataset.num_classes,
                embedding_dim=256,
                pretrained=True,
                pooling=args.pooling,)
    logging.info('\nNetwork config: \n%s'%(wetr))
    param_groups = wetr.get_param_groups()
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
    wetr.to(device)
    par.to(device)

    mask_size = int(cfg.dataset.crop_size // 16)
    infer_size = int((cfg.dataset.crop_size * max(cfg.cam.scales)) // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    attn_mask_infer = get_mask_by_radius(h=infer_size, w=infer_size, radius=args.radius)
    if args.local_rank==0:
        writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
        dummy_input = torch.rand(1, 3, 384, 384).to(device)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)
    if torch.cuda.is_available():
        wetr = DistributedDataParallel(
            wetr, 
            device_ids=[args.local_rank],
            find_unused_parameters=False, 
            broadcast_buffers=False)

    train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    bkg_cls = torch.ones(size=(cfg.train.samples_per_gpu, 1))

    # Threshold for switching from pseudo labels to affinity labels
    # Scaled down from COCO's 15000 based on smaller dataset
    aff_label_switch_iter = int(cfg.train.max_iters * 0.5)

    # Best checkpoint tracking (use -1.0 to ensure first validation saves a checkpoint)
    best_cls_score = -1.0
    best_ckpt_path = os.path.join(cfg.work_dir.ckpt_dir, "best.pth")

    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)

        # Forward pass through model
        cls, segs, attns, attn_pred = wetr(inputs, seg_detach=args.seg_detach)
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_labels)

        # Phase 1: Classification-only training (before cam_iters)
        # Skip expensive CAM/PAR refinement since seg_loss and aff_loss are zeroed anyway
        if n_iter <= cfg.train.cam_iters:
            # Add dummy terms (multiplied by 0) to ensure all parameters receive gradients
            # This is required for DDP with find_unused_parameters=False
            dummy_seg = 0.0 * segs.sum()
            dummy_attn = 0.0 * attn_pred.sum()
            loss = cls_loss + dummy_seg + dummy_attn
            # Placeholder values for logging
            seg_loss = torch.tensor(0.0, device=device)
            aff_loss = torch.tensor(0.0, device=device)
            pos_count = torch.tensor(0, device=device)
            neg_count = torch.tensor(0, device=device)
            # Placeholders for tensorboard visualization (will be skipped in logging block)
            valid_cam = None
            aff_cam = None
            refined_aff_cam = None
            pseudo_label = None
            refined_pseudo_label = None
            refined_aff_label = None
        else:
            # Phase 2+: Full training with CAM/affinity refinement
            inputs_denorm = imutils.denormalize_img2(inputs.clone())
            
            cams, aff_mat = multi_scale_cam_with_aff_mat(wetr, inputs=inputs, scales=cfg.cam.scales)
            valid_cam, pseudo_label = cam_to_label(cams.detach(), cls_label=cls_labels, img_box=img_box, ignore_mid=True, cfg=cfg)

            valid_cam_resized = F.interpolate(valid_cam, size=(infer_size, infer_size), mode='bilinear', align_corners=False)

            # Affinity refinement only needed after aff_label_switch_iter
            if n_iter > aff_label_switch_iter:
                # Full affinity-based refinement path
                aff_cam_l = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.low_thre)
                aff_cam_l = F.interpolate(aff_cam_l, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                aff_cam_h = propagte_aff_cam_with_bkg(valid_cam_resized, aff=aff_mat.detach().clone(), mask=attn_mask_infer, cls_labels=cls_labels, bkg_score=cfg.cam.high_thre)
                aff_cam_h = F.interpolate(aff_cam_h, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

                bkg_cls = bkg_cls.to(cams.device)
                _cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

                refined_aff_cam_l = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_l, labels=_cls_labels, img_box=img_box)
                refined_aff_label_l = refined_aff_cam_l.argmax(dim=1)
                refined_aff_cam_h = refine_cams_with_cls_label(par, inputs_denorm, cams=aff_cam_h, labels=_cls_labels, img_box=img_box)
                refined_aff_label_h = refined_aff_cam_h.argmax(dim=1)

                aff_cam = aff_cam_l[:,1:]
                refined_aff_cam = refined_aff_cam_l[:,1:,]
                refined_aff_label = refined_aff_label_h.clone()
                refined_aff_label[refined_aff_label_h == 0] = cfg.dataset.ignore_index
                refined_aff_label[(refined_aff_label_h + refined_aff_label_l) == 0] = 0
                refined_aff_label = ignore_img_box(refined_aff_label, img_box=img_box, ignore_index=cfg.dataset.ignore_index)
            else:
                # Before aff_label_switch_iter: use pseudo labels from CAM refinement only
                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=cams, cls_labels=cls_labels, cfg=cfg, img_box=img_box)
                refined_aff_label = refined_pseudo_label
                # Placeholders for tensorboard
                aff_cam = valid_cam  # Use valid_cam as placeholder
                refined_aff_cam = valid_cam

            # Store refined_pseudo_label for logging if not already set
            if n_iter > aff_label_switch_iter:
                refined_pseudo_label = refined_aff_label  # For logging consistency

            aff_label = cams_to_affinity_label(refined_aff_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
            aff_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

            segs = F.interpolate(segs, size=refined_aff_label.shape[1:], mode='bilinear', align_corners=False)
            seg_loss = get_seg_loss(segs, refined_aff_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

            loss = 1.0 * cls_loss + 0.1 * seg_loss + 0.1 * aff_loss


        avg_meter.add({'cls_loss': cls_loss.item(), 'seg_loss': seg_loss.item(), 'aff_loss': aff_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter+1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            # Phase 1 (before cam_iters): simplified logging without CAM visualizations
            if n_iter <= cfg.train.cam_iters:
                seg_mAcc = 0.0  # No pseudo-labels computed yet
                if args.local_rank==0:
                    logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f (cls-only phase)"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('cls_loss')))
                    avg_meter.pop('aff_loss')  # Clear placeholder
                    avg_meter.pop('seg_loss')  # Clear placeholder
                    
                    # Only log attention maps during early phase
                    _attns_detach = [a.detach() for a in attns]
                    _attns_detach.append(attn_pred.detach())
                    grid_attns = imutils.tensorboard_attn2(attns=_attns_detach, n_row=cfg.train.samples_per_gpu)
                    
                    _imgs = imutils.denormalize_img(inputs.clone())
                    grid_imgs = torchvision.utils.make_grid(tensor=_imgs, nrow=2)
                    writer.add_image("train/images", grid_imgs, global_step=n_iter)
                    writer.add_image("attns/top_stages_case0", grid_attns[0], global_step=n_iter)
                    writer.add_image("attns/top_stages_case1", grid_attns[1], global_step=n_iter)
                    writer.add_image("attns/top_stages_case2", grid_attns[2], global_step=n_iter)
                    writer.add_image("attns/top_stages_case3", grid_attns[3], global_step=n_iter)
                    writer.add_image("attns/last_stage_case0", grid_attns[4], global_step=n_iter)
                    writer.add_image("attns/last_stage_case1", grid_attns[5], global_step=n_iter)
                    writer.add_image("attns/last_stage_case2", grid_attns[6], global_step=n_iter)
                    writer.add_image("attns/last_stage_case3", grid_attns[7], global_step=n_iter)
                    writer.add_scalars('train/loss', {"cls_loss": cls_loss.item()}, global_step=n_iter)
            else:
                # Phase 2+: full logging with CAM visualizations
                preds = torch.argmax(segs,dim=1,).cpu().numpy().astype(np.int16)
                gts = pseudo_label.cpu().numpy().astype(np.int16)
                refined_gts = refined_pseudo_label.cpu().numpy().astype(np.int16)
                aff_gts = refined_aff_label.cpu().numpy().astype(np.int16)

                seg_mAcc = (preds==gts).sum()/preds.size

                grid_imgs, grid_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=valid_cam)
                _, grid_aff_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=aff_cam)
                _, grid_ref_aff_cam = imutils.tensorboard_image(imgs=inputs.clone(), cam=refined_aff_cam)

                _attns_detach = [a.detach() for a in attns]
                _attns_detach.append(attn_pred.detach())
                grid_attns = imutils.tensorboard_attn2(attns=_attns_detach, n_row=cfg.train.samples_per_gpu)

                grid_labels = imutils.tensorboard_label(labels=gts)
                grid_preds = imutils.tensorboard_label(labels=preds)
                grid_refined_gt = imutils.tensorboard_label(labels=refined_gts)
                grid_aff_gt = imutils.tensorboard_label(labels=aff_gts)

                if args.local_rank==0:
                    logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, aff_loss: %.4f, pseudo_seg_loss %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('aff_loss'), avg_meter.pop('seg_loss'), seg_mAcc))

                    writer.add_image("train/images", grid_imgs, global_step=n_iter)
                    writer.add_image("train/preds", grid_preds, global_step=n_iter)
                    writer.add_image("train/pseudo_gts", grid_labels, global_step=n_iter)
                    writer.add_image("train/pseudo_ref_gts", grid_refined_gt, global_step=n_iter)
                    writer.add_image("train/aff_gts", grid_aff_gt, global_step=n_iter)
                    writer.add_image("cam/valid_cams", grid_cam, global_step=n_iter)
                    writer.add_image("cam/aff_cams", grid_aff_cam, global_step=n_iter)
                    writer.add_image("cam/refined_aff_cams", grid_ref_aff_cam, global_step=n_iter)

                    writer.add_image("attns/top_stages_case0", grid_attns[0], global_step=n_iter)
                    writer.add_image("attns/top_stages_case1", grid_attns[1], global_step=n_iter)
                    writer.add_image("attns/top_stages_case2", grid_attns[2], global_step=n_iter)
                    writer.add_image("attns/top_stages_case3", grid_attns[3], global_step=n_iter)

                    writer.add_image("attns/last_stage_case0", grid_attns[4], global_step=n_iter)
                    writer.add_image("attns/last_stage_case1", grid_attns[5], global_step=n_iter)
                    writer.add_image("attns/last_stage_case2", grid_attns[6], global_step=n_iter)
                    writer.add_image("attns/last_stage_case3", grid_attns[7], global_step=n_iter)

                    writer.add_scalars('train/loss', {"seg_loss": seg_loss.item(), "cls_loss": cls_loss.item()}, global_step=n_iter)
                    writer.add_scalar('count/pos_count', pos_count.item(), global_step=n_iter)
                    writer.add_scalar('count/neg_count', neg_count.item(), global_step=n_iter)


        if (n_iter+1) % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth"%(n_iter+1))
            if args.local_rank==0:
                logging.info('Validating...')
                torch.save(wetr.state_dict(), ckpt_name)
            cls_score, seg_score, cam_score, aff_score = validate(model=wetr, data_loader=val_loader, cfg=cfg, device=device)
            if args.local_rank==0:
                logging.info("=== Validation Scores ===")
                logging.info("val cls score: %.6f"%(cls_score))
                logging.info("(seg/cam/aff scores N/A - no GT masks for KITS)")

                # Track best checkpoint based on classification score
                if cls_score > best_cls_score:
                    best_cls_score = cls_score
                    torch.save(wetr.state_dict(), best_ckpt_path)
                    logging.info("New best checkpoint saved: cls_score=%.6f" % cls_score)

    # Post-training: Generate attention maps using best checkpoint
    if args.local_rank == 0:
        logging.info("Training complete. Generating attention maps...")
        generate_attention_maps_after_training(cfg, best_ckpt_path, device, args)

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    # Get local_rank from environment variable if not provided via command line
    # (newer PyTorch/torchrun uses env var instead of --local_rank argument)
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size
    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    if args.local_rank == 0:
        setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)
