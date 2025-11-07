from __future__ import print_function, division
import sys
sys.path.append('core')
import os
import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from raft_stereo import RAFTStereo, autocast
import stereo_datasets as datasets
from utils.utils import InputPadder
import csv
import cv2
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate_eth3d(model, iters=32, mixed_prec=False):
    """ Peform validation using the ETH3D (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5
        out = (epe_flattened > 1.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation ETH3D: EPE %f, D1 %f" % (epe, d1))
    return {'eth3d-epe': epe, 'eth3d-d1': d1}


@torch.no_grad()
def validate_kitti(model, iters=32, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, elapsed_list = [], [], []
    for val_id in range(len(val_dataset)):
        imagefile, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        left_image_path = imagefile[0]
        filename = os.path.basename(left_image_path)
        
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()

        if val_id > 50:
            elapsed_list.append(end-start)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = valid_gt.flatten() >= 0.5

        out = (epe_flattened > 3.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        if val_id < 9 or (val_id+1)%10 == 0:
            logging.info(f"KITTI Iter {val_id+1} [{filename}] out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}. Runtime: {format(end-start, '.3f')}s ({format(1/(end-start), '.2f')}-FPS)")
        epe_list.append(epe_flattened[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    avg_runtime = np.mean(elapsed_list)

    print(f"Validation KITTI: EPE {epe}, D1 {d1}, {format(1/avg_runtime, '.2f')}-FPS ({format(avg_runtime, '.3f')}s)")
    return {'kitti-epe': epe, 'kitti-d1': d1}

# 确保在 evaluate_stereo_v2.py 文件顶部有这个导入

@torch.no_grad()
def validate_mydataset(model, iters=32, mixed_prec=False, output_csv_path='iraft_results.csv', visualization_dir='output'):
    """ 
    Peform validation using a custom dataset.
    This function calculates detailed metrics, saves them to a CSV file,
    and saves composite visualization images of the results.
    """
    model.eval()
    aug_params = {}
    val_dataset = datasets.MyDataSet(aug_params) 
    torch.backends.cudnn.benchmark = True

    if visualization_dir:
        os.makedirs(visualization_dir, exist_ok=True)
        logging.info(f"Visualization images will be saved to: {visualization_dir}")
        
    results_data = []
    epe_list, out_list_d1, elapsed_list = [], [], []

    for val_id in range(len(val_dataset)):
    
        image_files, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        
        image1_for_vis = image1.clone()
        
        left_image_path = image_files[0]
        filename = os.path.basename(left_image_path)

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        inference_size = f"{image1.shape[2]}x{image1.shape[3]}"

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        torch.cuda.reset_peak_memory_stats()

        with autocast(enabled=mixed_prec):
            start = time.time()
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
            end = time.time()
        
        inference_time_ms = (end - start) * 1000
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        if val_id > 50:
            elapsed_list.append(end-start)

        flow_pr = padder.unpad(flow_pr).cpu() # 保持为Tensor
        
        # 确保所有张量的维度和形状统一
        flow_gt = flow_gt.squeeze()
        flow_pr = flow_pr.squeeze()
        valid_gt = valid_gt.squeeze()

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        
        # ==================== 新增：可视化代码块 ====================
        if visualization_dir:
            try:
                # 1. 准备左原图 (RGB -> BGR, Tensor -> NumPy)
                img_vis_left = image1_for_vis.permute(1, 2, 0).numpy().astype(np.uint8)
                img_vis_left_bgr = cv2.cvtColor(img_vis_left, cv2.COLOR_RGB2BGR)

                # 2. 准备预测视差图和真值视差图 (转换为伪彩色)
                disp_pr_np = np.abs(flow_pr.numpy())
                disp_gt_np = np.abs(flow_gt.numpy())

                # 统一颜色映射的范围，使两图可比
                vmax = max(np.max(disp_pr_np), np.max(disp_gt_np))
                if vmax <= 0: vmax = 1.0 # 避免除以零

                # 归一化到0-255并应用JET colormap
                disp_pr_color = cv2.applyColorMap((disp_pr_np / vmax * 255).astype(np.uint8), cv2.COLORMAP_JET)
                disp_gt_color = cv2.applyColorMap((disp_gt_np / vmax * 255).astype(np.uint8), cv2.COLORMAP_JET)

                # 在真值视差图上，将无效区域涂黑以便区分
                valid_mask_np = (valid_gt.numpy() >= 0.5)
                disp_gt_color[~valid_mask_np] = 0

                # 3. 水平拼接三张图像
                vis_image = cv2.hconcat([img_vis_left_bgr, disp_pr_color, disp_gt_color])

                # 4. 保存拼接后的图像
                save_path = os.path.join(visualization_dir, filename)
                cv2.imwrite(save_path, vis_image)

            except Exception as e:
                import traceback
                logging.warning(f"Could not save visualization for {filename}. Error: {e}\n{traceback.format_exc()}")
     

        epe = torch.abs(flow_pr - flow_gt)
        epe_flattened = epe.flatten()
        
        if valid_gt is not None:
            val = valid_gt.flatten() >= 0.5
        else:
            val = torch.ones(epe_flattened.shape, dtype=torch.bool)

        if torch.sum(val) == 0:
            logging.warning(f"Skipping {filename} due to no valid GT pixels.")
            continue

        image_epe = epe_flattened[val].mean().item()

        bp1 = 100 * ((epe_flattened > 1.0)[val].float().mean().item())
        bp2 = 100 * ((epe_flattened > 2.0)[val].float().mean().item())
        bp3 = 100 * ((epe_flattened > 3.0)[val].float().mean().item())
        bp5 = 100 * ((epe_flattened > 5.0)[val].float().mean().item())
        
        out_d1_pixels = (epe_flattened > 3.0)[val].cpu().numpy()
        out_list_d1.append(out_d1_pixels)
        epe_list.append(image_epe)

        log_msg = (f"MyDataset Iter {val_id+1}/{len(val_dataset)} [{filename}] - "
                   f"EPE: {image_epe:.4f}, BP-1: {bp1:.4f}, BP-2: {bp2:.4f}, D1(BP-3): {bp3:.4f}, BP-5: {bp5:.4f}, "
                   f"Time: {inference_time_ms:.2f}ms, Mem: {peak_memory_mb:.2f}MB")
        logging.info(log_msg)
        
        current_result = {
            'filename': filename, 'inference_size': inference_size, 'BP-1': f"{bp1:.4f}",
            'BP-2': f"{bp2:.4f}", 'BP-3': f"{bp3:.4f}", 'BP-5': f"{bp5:.4f}", 'EPE': f"{image_epe:.4f}",
            'D1': f"{bp3:.4f}", 'inference_time_ms': f"{inference_time_ms:.4f}", 'peak_memory_mb': f"{peak_memory_mb:.4f}"
        }
        results_data.append(current_result)

    if output_csv_path:
        logging.info(f"Saving results to {output_csv_path}")
        fieldnames = ['filename', 'inference_size', 'BP-1', 'BP-2', 'BP-3', 'BP-5', 'EPE', 'D1', 'inference_time_ms', 'peak_memory_mb']
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_data)
        logging.info("CSV file saved successfully.")

    avg_epe = np.mean(epe_list) if epe_list else 0
    if out_list_d1:
        out_list_d1 = np.concatenate(out_list_d1)
        avg_d1 = 100 * np.mean(out_list_d1)
    else:
        avg_d1 = 0.0

    avg_runtime = np.mean(elapsed_list) if elapsed_list else 0
    avg_fps = 1 / avg_runtime if avg_runtime > 0 else 0

    print(f"Validation MyDataset Summary: EPE {avg_epe:.4f}, D1 {avg_d1:.4f}, {avg_fps:.2f}-FPS ({avg_runtime*1000:.2f}ms)")
    return {'mydataset-epe': avg_epe, 'mydataset-d1': avg_d1}

# @torch.no_grad()
# def validate_mydataset(model, iters=32, mixed_prec=False, output_csv_path='iraft_results.csv',visualization_dir='output'): # 建议改个默认文件名
#     """ 
#     Peform validation using a custom dataset.
#     This function calculates detailed metrics and saves them to a CSV file.
#     The D1 calculation logic is aligned with the standard KITTI (macro-average) method.
#     """
#     model.eval()
#     aug_params = {}
#     # 修正：MyDataSet 不应该需要 image_set 参数
#     val_dataset = datasets.MyDataSet(aug_params) 
#     torch.backends.cudnn.benchmark = True
#      # --- 3. 如果指定了可视化目录，则创建它 ---
#     if visualization_dir:
#         os.makedirs(visualization_dir, exist_ok=True)
#         logging.info(f"Visualization images will be saved to: {visualization_dir}")
        
#     results_data = []
    
#     # 修正：采用和 validate_kitti 一样的逻辑来收集数据
#     epe_list, out_list_d1, elapsed_list = [], [], []

#     for val_id in range(len(val_dataset)):
    
#         image_files, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        
#         # 保存一份未被修改的、用于可视化的左图张量
#         image1_for_vis = image1.clone()
        
#         left_image_path = image_files[0]
#         filename = os.path.basename(left_image_path)

#         image1 = image1[None].cuda()
#         image2 = image2[None].cuda()
        
#         inference_size = f"{image1.shape[2]}x{image1.shape[3]}"

#         padder = InputPadder(image1.shape, divis_by=32)
#         image1, image2 = padder.pad(image1, image2)

#         torch.cuda.reset_peak_memory_stats()

#         with autocast(enabled=mixed_prec):
#             start = time.time()
#             _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
#             end = time.time()
        
#         inference_time_ms = (end - start) * 1000
#         peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
#         if val_id > 50:
#             elapsed_list.append(end-start)

#         flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
#         assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        
#         epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
#         epe_flattened = epe.flatten()
        
#         # 检查是否有有效的真值像素，防止计算 mean of empty tensor
#         if valid_gt is not None:
#             val = valid_gt.flatten() >= 0.5
#         else: # 如果没有提供valid_gt，则假设所有像素都有效
#             val = torch.ones(epe_flattened.shape, dtype=torch.bool)

#         if torch.sum(val) == 0:
#             logging.warning(f"Skipping {filename} due to no valid GT pixels.")
#             continue # 如果没有有效像素，跳过这张图

#         image_epe = epe_flattened[val].mean().item()

#         # 计算不同阈值下的Bad Pixel百分比 (仅用于单张图的CSV记录)
#         bp1 = 100 * ((epe_flattened > 1.0)[val].float().mean().item())
#         bp2 = 100 * ((epe_flattened > 2.0)[val].float().mean().item())
#         bp3 = 100 * ((epe_flattened > 3.0)[val].float().mean().item())
#         bp5 = 100 * ((epe_flattened > 5.0)[val].float().mean().item())
        
#         # 修正：为总D1计算收集原始坏点数据
#         out_d1_pixels = (epe_flattened > 3.0)[val].cpu().numpy()
#         out_list_d1.append(out_d1_pixels)
#         epe_list.append(image_epe)

#         log_msg = (f"MyDataset Iter {val_id+1}/{len(val_dataset)} [{filename}] - "
#                    f"EPE: {image_epe:.4f}, BP-1: {bp1:.4f}, BP-2: {bp2:.4f}, D1(BP-3): {bp3:.4f}, BP-5: {bp5:.4f}, "
#                    f"Time: {inference_time_ms:.2f}ms, Mem: {peak_memory_mb:.2f}MB")
#         logging.info(log_msg)
        
#         current_result = {
#             'filename': filename, 'inference_size': inference_size, 'BP-1': f"{bp1:.4f}",
#             'BP-2': f"{bp2:.4f}", 'BP-3': f"{bp3:.4f}", 'BP-5': f"{bp5:.4f}", 'EPE': f"{image_epe:.4f}",
#             'D1': f"{bp3:.4f}", 'inference_time_ms': f"{inference_time_ms:.4f}", 'peak_memory_mb': f"{peak_memory_mb:.4f}"
#         }
#         results_data.append(current_result)

#     if output_csv_path:

#         logging.info(f"Saving results to {output_csv_path}")
#         fieldnames = ['filename', 'inference_size', 'BP-1', 'BP-2', 'BP-3', 'BP-5', 'EPE', 'D1', 'inference_time_ms', 'peak_memory_mb']
#         with open(output_csv_path, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(results_data)
#         logging.info("CSV file saved successfully.")

#     # 修正：使用宏平均计算最终的平均指标
#     avg_epe = np.mean(epe_list)
#     out_list_d1 = np.concatenate(out_list_d1)
#     avg_d1 = 100 * np.mean(out_list_d1)

#     avg_runtime = np.mean(elapsed_list) if elapsed_list else 0
#     avg_fps = 1 / avg_runtime if avg_runtime > 0 else 0

#     print(f"Validation MyDataset Summary: EPE {avg_epe:.4f}, D1 {avg_d1:.4f}, {avg_fps:.2f}-FPS ({avg_runtime*1000:.2f}ms)")
#     return {'mydataset-epe': avg_epe, 'mydataset-d1': avg_d1}

# @torch.no_grad()
# def validate_mydataset(model, iters=32, mixed_prec=False, output_csv_path='kitti_results.csv'):
#     """ 
#     Peform validation using the KITTI-2015 (train) split.
#     This function has been modified to:
#     1. Calculate additional Bad Pixel (BP) metrics.
#     2. Measure inference time and peak GPU memory usage.
#     3. Save all results to a specified CSV file.
#     """
#     model.eval()
#     aug_params = {}
#     val_dataset = datasets.MyDataSet(aug_params, image_set='training')
#     torch.backends.cudnn.benchmark = True

#     # 初始化用于存储所有结果的列表
#     results_data = []
#     # 初始化用于计算最终平均值的列表
#     epe_list, d1_list, bp1_list, bp2_list, bp3_list, bp5_list = [], [], [], [], [], []
#     elapsed_list = []

#     for val_id in range(len(val_dataset)):
#         # 从数据集中获取文件名、图像和真值
#         # 注意: StereoDataset的__getitem__需要返回文件名信息
#         # 我们假设它存储在 val_dataset.image_list 中
#         image_files, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        
#         # 提取主文件名用于CSV记录
#         left_image_path = image_files[0]
#         filename = os.path.basename(left_image_path)

#         image1 = image1[None].cuda()
#         image2 = image2[None].cuda()
        
#         # 记录推理前的图像尺寸
#         inference_size = f"{image1.shape[2]}x{image1.shape[3]}"

#         padder = InputPadder(image1.shape, divis_by=32)
#         image1, image2 = padder.pad(image1, image2)

#         # 重置CUDA峰值内存统计数据
#         torch.cuda.reset_peak_memory_stats()

#         with autocast(enabled=mixed_prec):
#             start = time.time()
#             _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
#             end = time.time()
        
#         # 性能指标
#         inference_time_ms = (end - start) * 1000
#         peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
#         if val_id > 50: # 预热后记录时间
#             elapsed_list.append(end-start)

#         flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
#         assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        
#         # 计算EPE (End-Point-Error)
#         epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()
#         epe_flattened = epe.flatten()
#         val = valid_gt.flatten() >= 0.5 # 仅在有效的真值区域进行评估

#         # 计算各项精度指标
#         image_epe = epe_flattened[val].mean().item()

#         # 计算不同阈值下的Bad Pixel百分比
#         bp1 = 100 * ((epe_flattened > 1.0)[val].float().mean().item())
#         bp2 = 100 * ((epe_flattened > 2.0)[val].float().mean().item())
#         bp3 = 100 * ((epe_flattened > 3.0)[val].float().mean().item()) # D1与BP-3通常使用相同阈值
#         bp5 = 100 * ((epe_flattened > 5.0)[val].float().mean().item())

#         # 打印单张图片的评估信息
#         log_msg = (f"KITTI Iter {val_id+1}/{len(val_dataset)} [{filename}] - "
#                    f"EPE: {image_epe:.4f}, BP-1: {bp1:.4f}, BP-2: {bp2:.4f}, D1(BP-3): {bp3:.4f}, BP-5: {bp5:.4f}, "
#                    f"Time: {inference_time_ms:.2f}ms, Mem: {peak_memory_mb:.2f}MB")
#         logging.info(log_msg)
        
#         # 将当前图片的结果存入字典
#         current_result = {
#             'filename': filename,
#             'inference_size': inference_size,
#             'BP-1': f"{bp1:.4f}",
#             'BP-2': f"{bp2:.4f}",
#             'BP-3': f"{bp3:.4f}",
#             'BP-5': f"{bp5:.4f}",
#             'EPE': f"{image_epe:.4f}",
#             'D1': f"{bp3:.4f}", # 按照图片格式，D1列与BP-3值相同
#             'inference_time_ms': f"{inference_time_ms:.4f}",
#             'peak_memory_mb': f"{peak_memory_mb:.4f}"
#         }
#         results_data.append(current_result)

#         # 收集数据用于计算最终的平均值
#         epe_list.append(image_epe)
#         bp1_list.append(bp1)
#         bp2_list.append(bp2)
#         bp3_list.append(bp3)
#         bp5_list.append(bp5)

#     # 将所有结果写入CSV文件
#     if output_csv_path:
#         logging.info(f"Saving results to {output_csv_path}")
#         fieldnames = ['filename', 'inference_size', 'BP-1', 'BP-2', 'BP-3', 'BP-5', 'EPE', 'D1', 'inference_time_ms', 'peak_memory_mb']
#         with open(output_csv_path, 'w', newline='') as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerows(results_data)
#         logging.info("CSV file saved successfully.")

#     # 计算并打印数据集的平均指标
#     avg_epe = np.mean(epe_list)
#     avg_d1 = np.mean(bp3_list) # 平均D1(BP-3)
#     avg_runtime = np.mean(elapsed_list) if elapsed_list else 0
#     avg_fps = 1 / avg_runtime if avg_runtime > 0 else 0

#     print(f"Validation KITTI Summary: EPE {avg_epe:.4f}, D1(BP-3) {avg_d1:.4f}, {avg_fps:.2f}-FPS ({avg_runtime*1000:.2f}ms)")
#     return {'kitti-epe': avg_epe, 'kitti-d1': avg_d1}
@torch.no_grad()
def validate_things(model, iters=32, mixed_prec=False):
    """ Peform validation using the FlyingThings3D (TEST) split """
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(dstype='frames_finalpass', things_test=True)

    out_list, epe_list = [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe = epe.flatten()
        val = (valid_gt.flatten() >= 0.5) & (flow_gt.abs().flatten() < 192)

        out = (epe > 1.0)
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print("Validation FlyingThings: %f, %f" % (epe, d1))
    return {'things-epe': epe, 'things-d1': d1}


@torch.no_grad()
def validate_middlebury(model, iters=32, split='F', mixed_prec=False):
    """ Peform validation using the Middlebury-V3 dataset """
    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt)**2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (valid_gt.reshape(-1) >= -0.5) & (flow_gt[0].reshape(-1) > -1000)

        out = (epe_flattened > 2.0)
        image_out = out[val].float().mean().item()
        image_epe = epe_flattened[val].mean().item()
        logging.info(f"Middlebury Iter {val_id+1} out of {len(val_dataset)}. EPE {round(image_epe,4)} D1 {round(image_out,4)}")
        epe_list.append(image_epe)
        out_list.append(image_out)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)

    print(f"Validation Middlebury{split}: EPE {epe}, D1 {d1}")
    return {f'middlebury{split}-epe': epe, f'middlebury{split}-d1': d1}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default=None)
    parser.add_argument('--dataset', help="dataset for evaluation", required=True, choices=["eth3d", "kitti", "things","custom"] + [f"middlebury_{s}" for s in 'FHQ'])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=64, help='number of flow-field updates during forward pass')

    # Architecure choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors. 
    use_mixed_precision = args.corr_implementation.endswith("_cuda")

    if args.dataset == 'eth3d':
        validate_eth3d(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset == 'kitti':
        validate_kitti(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
        
    elif args.dataset == 'custom':
        validate_mydataset(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)

    elif args.dataset in [f"middlebury_{s}" for s in 'FHQ']:
        validate_middlebury(model, iters=args.valid_iters, split=args.dataset[-1], mixed_prec=use_mixed_precision)

    elif args.dataset == 'things':
        validate_things(model, iters=args.valid_iters, mixed_prec=use_mixed_precision)
