import torch
from torch.utils.cpp_extension import load_inline

def get_conv2d_index(
    grid_thw,
    spatial_merge_size,
    kernel_size,
    stride,
    padding_left=2,
):
    grid_thw = grid_thw.cpu()
    repeats = grid_thw[:, 0]
    height_width = torch.repeat_interleave(grid_thw[:, 1:], repeats, dim=0)
    num_grids_per_img = height_width[:, 0] * height_width[:, 1]
    
    expand_h = (height_width[:, 0] + padding_left - kernel_size) // stride + 1
    expand_w = (height_width[:, 1] + padding_left - kernel_size) // stride + 1
    # num_expand_patch_per_img = num_units_per_img // (spatial_merge_size ** 2) * (kernel_size ** 2)
    num_expand_patch_per_img = expand_h * expand_w * (kernel_size ** 2)
    
    total_grids = grid_thw.prod(dim=1).sum().item()
    # index_len = total_grids // (spatial_merge_size ** 2) * (kernel_size ** 2)
    index_len = num_expand_patch_per_img.sum().item()
    
    pre_cumsum = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(num_expand_patch_per_img, dim=0)[:-1]], dim=0)
    
    height_width_flat = torch.repeat_interleave(height_width, num_expand_patch_per_img, dim=0)
    pre_cumsum_flat = torch.repeat_interleave(pre_cumsum, num_expand_patch_per_img, dim=0)
    pre_cumsum_origin_patch_flat = torch.repeat_interleave(
        torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(num_grids_per_img, dim=0)[:-1]]),
        num_expand_patch_per_img,
        dim=0
    )
    
    device = pre_cumsum_flat.device
    arange_i = torch.arange(index_len, device=device, dtype=torch.int64)

    # 2. 计算 off
    # 原逻辑: off = i - pre_cumsum_flat[i]
    off = arange_i - pre_cumsum_flat.long()

    # 3. 计算 patch_every_row
    # 原逻辑: height_width_flat[i,1] // spatial_merge_size * (kernel_size**2)
    # 提取 width 列
    width_flat = height_width_flat[:, 1].long()
    height_flat = height_width_flat[:, 0].long()
    k2 = kernel_size ** 2

    patch_every_row = (width_flat // spatial_merge_size) * k2

    # 4. 计算 ker_idx_row 和 ker_idx_col
    # 原逻辑: ker_idx_row = off // patch_every_row
    ker_idx_row = off // patch_every_row

    # 原逻辑: ker_idx_col = off % patch_every_row // (kernel_size**2)
    ker_idx_col = (off % patch_every_row) // k2

    # 5. 计算 local indices
    # 原逻辑: local_off = off % (kernel_size**2)
    local_off = off % k2
    local_row = local_off // kernel_size
    local_col = local_off % kernel_size

    # 6. 计算 pos_row 和 pos_col 并进行边界截断 (clamp/min)
    # 原逻辑: min(..., height - 1)
    raw_pos_row = ker_idx_row * stride + local_row - padding_left
    # pos_row = torch.minimum(raw_pos_row, height_flat - 1)
    pos_row = torch.clamp(raw_pos_row, 0)

    raw_pos_col = ker_idx_col * stride + local_col - padding_left
    # pos_col = torch.minimum(raw_pos_col, width_flat - 1)
    pos_col = torch.clamp(raw_pos_col, 0)

    # 7. 计算最终的 conv2d_index
    # 原逻辑: pre_cumsum_origin_patch_flat[i] + pos_row * width + pos_col
    conv2d_index = (pre_cumsum_origin_patch_flat.long() + 
                    pos_row * width_flat + 
                    pos_col)

    # 如果需要转回 int32
    conv2d_index = conv2d_index.to(torch.int32)
    
    return conv2d_index

cpp_source = """
#include <torch/extension.h>
#include <vector>

// C++ 实现：计算 Conv2d 的 gather 索引
// 逻辑已更新：适配新的 expand_h/w 计算公式以及 padding_left 参数
torch::Tensor get_conv2d_index_cpp(
    torch::Tensor grid_thw,      // [N, 3] -> (T, H, W)
    int spatial_merge_size,      // 保留接口一致性，但计算逻辑已改为基于 stride/kernel
    int kernel_size,
    int stride,
    int padding_left             // 对应 Python 中的 padding_left
) {
    grid_thw = grid_thw.cpu();
    auto grid_acc = grid_thw.accessor<int64_t, 2>();
    int num_configs = grid_thw.size(0);
    int64_t k2 = kernel_size * kernel_size;

    // -----------------------------------------------------------
    // 1. 预计算总索引长度 (用来分配内存)
    // -----------------------------------------------------------
    int64_t total_indices = 0;
    for(int i = 0; i < num_configs; ++i) {
        int64_t t = grid_acc[i][0];
        int64_t h = grid_acc[i][1];
        int64_t w = grid_acc[i][2];
        
        // Python 逻辑对应: 
        // expand_h = (h + padding_left - kernel_size) // stride + 1
        int64_t out_h = (h + padding_left - kernel_size) / stride + 1;
        int64_t out_w = (w + padding_left - kernel_size) / stride + 1;
        
        // 简单的安全检查，防止参数极不合理导致负数
        if (out_h < 0) out_h = 0;
        if (out_w < 0) out_w = 0;
        
        total_indices += t * (out_h * out_w) * k2;
    }

    // 2. 分配输出 Tensor
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor indices = torch::empty({total_indices}, options);
    auto idx_acc = indices.accessor<int, 1>();

    // -----------------------------------------------------------
    // 3. 主循环生成索引
    // -----------------------------------------------------------
    int64_t global_idx_ptr = 0;       
    int64_t input_pixel_offset = 0;   

    for(int i = 0; i < num_configs; ++i) {
        int64_t t = grid_acc[i][0];
        int64_t h = grid_acc[i][1];
        int64_t w = grid_acc[i][2];
        int64_t pixels_per_frame = h * w;
        
        // 计算当前图片的输出网格大小
        int64_t out_rows = (h + padding_left - kernel_size) / stride + 1;
        int64_t out_cols = (w + padding_left - kernel_size) / stride + 1;
        if (out_rows < 0) out_rows = 0;
        if (out_cols < 0) out_cols = 0;

        // 遍历每一帧 (T)
        for(int frame = 0; frame < t; ++frame) {
            
            // 遍历输出 Patch (Output Height)
            for(int row = 0; row < out_rows; ++row) {
                // 遍历输出 Patch (Output Width)
                for(int col = 0; col < out_cols; ++col) {
                    
                    // 遍历卷积核 (Kernel Height)
                    for(int ky = 0; ky < kernel_size; ++ky) {
                        // 遍历卷积核 (Kernel Width)
                        for(int kx = 0; kx < kernel_size; ++kx) {
                            
                            // 坐标计算公式 (对应 Python):
                            // raw_pos = ker_idx * stride + local - padding_left
                            int64_t in_row = row * stride + ky - padding_left;
                            int64_t in_col = col * stride + kx - padding_left;
                            
                            // Replicate Padding 边界处理
                            // 必须限制在 [0, h-1] 和 [0, w-1] 之间，否则 gather 会越界
                            if (in_row < 0) in_row = 0;
                            else if (in_row >= h) in_row = h - 1;
                            
                            if (in_col < 0) in_col = 0;
                            else if (in_col >= w) in_col = w - 1;
                            
                            // 计算绝对索引
                            int64_t abs_index = input_pixel_offset + in_row * w + in_col;
                            
                            idx_acc[global_idx_ptr++] = (int)abs_index;
                        }
                    }
                }
            }
            // 处理完一帧，输入偏移量增加
            input_pixel_offset += pixels_per_frame;
        }
    }

    return indices;
}
"""

# ==========================================
# 编译并加载
# ==========================================
# 这一步会自动编译 C++ 代码，第一次运行可能需要几秒钟
cpp_module = load_inline(
    name='conv_utils_cpp',
    cpp_sources=cpp_source,
    functions=['get_conv2d_index_cpp'],
    verbose=False
)

# ==========================================
# Python 包装函数 (无缝衔接)
# ==========================================
def get_conv2d_index_cpp(
    grid_thw, 
    spatial_merge_size, 
    kernel_size, 
    stride, 
    padding_left=2
):
    """
    Args:
        grid_thw: Tensor [N, 3] (T, H, W)
        spatial_merge_size: int
        kernel_size: int
        stride: int
        padding: int
    Returns:
        indices: Tensor [total_length], int32
    """
    # 确保 grid_thw 是 CPU int64 (C++要求)
    if grid_thw.device.type != 'cpu':
        grid_thw = grid_thw.cpu()
    grid_thw = grid_thw.to(torch.int64)
    
    if grid_thw.prod(dim=-1).sum() == 0:
        return torch.empty((0,), dtype=torch.int32)
        
    # 调用 C++
    indices = cpp_module.get_conv2d_index_cpp(
        grid_thw, 
        spatial_merge_size, 
        kernel_size, 
        stride, 
        padding_left
    )
    
    return indices

def get_spatial_unit_conv2d_index_cpp(
    grid_thw, 
    spatial_merge_size, 
    kernel_size=4, 
    stride=2, 
    padding_left=2
):
    assert kernel_size % spatial_merge_size == 0, "Kernel size must be multiple of spatial merge size"
    assert stride % spatial_merge_size == 0, "Stride must be multiple of spatial merge size"
    assert padding_left % spatial_merge_size == 0, "Padding must be multiple of spatial merge size"
    if grid_thw.device.type != 'cpu':
        grid_thw = grid_thw.cpu()
    grid_thw = grid_thw.to(torch.int64)
    
    if grid_thw.prod(dim=-1).sum() == 0:
        return torch.empty((0,), dtype=torch.int32)
    
    # T, H // spatial_merge_size, W // spatial_merge_size
    grid_thw[:, 1] = grid_thw[:, 1] // spatial_merge_size
    grid_thw[:, 2] = grid_thw[:, 2] // spatial_merge_size
    
    # indices = cpp_module.get_conv2d_index_cpp(
    indices = get_conv2d_index_cpp(
        grid_thw, 
        1,  # spatial_merge_size = 1
        kernel_size // spatial_merge_size,
        stride // spatial_merge_size,
        padding_left // spatial_merge_size
    )

    return indices

if __name__ == "__main__":
    spatial_merge_size = 2
    kernel_size = 4
    stride = 2

    # grid_thw = torch.tensor([[2, 4, 4], [3, 16, 16], [1, 4, 4]])
    grid_thw = torch.tensor([[1, 2, 2]])
    def naive_conv2d_index(
        grid_thw,
        spatial_merge_size,
        kernel_size,
        stride,
        padding=1,
    ):
        grid_thw = grid_thw.cpu()
        repeats = grid_thw[:, 0]
        height_width = torch.repeat_interleave(grid_thw[:, 1:], repeats, dim=0)
        num_units_per_img = height_width[:, 0] * height_width[:, 1]
        
        total_grids = grid_thw.prod(dim=1).sum().item()
        index_len = total_grids // (spatial_merge_size ** 2) * (kernel_size ** 2)
        
        num_expand_patch_per_img = num_units_per_img // (spatial_merge_size ** 2) * (kernel_size ** 2)
        
        pre_cumsum = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(num_expand_patch_per_img, dim=0)[:-1]], dim=0)
        
        height_width_flat = torch.repeat_interleave(height_width, num_expand_patch_per_img, dim=0)
        pre_cumsum_flat = torch.repeat_interleave(pre_cumsum, num_expand_patch_per_img, dim=0)
        pre_cumsum_origin_patch_flat = pre_cumsum_flat // (kernel_size ** 2) * (spatial_merge_size ** 2)
        
        conv2d_index = torch.empty((index_len,), dtype=torch.int32)
        
        for i in range(index_len):
            off = i - pre_cumsum_flat[i]
            patch_every_row = height_width_flat[i,1] // spatial_merge_size * (kernel_size**2)
            ker_idx_row = off // patch_every_row
            ker_idx_col = off % patch_every_row // (kernel_size**2)
            local_off = off % (kernel_size**2)
            
            # pos_row = min(ker_idx_row * stride + (local_off // kernel_size), height_width_flat[i,0] - 1)
            # pos_col = min(ker_idx_col * stride + (local_off % kernel_size), height_width_flat[i,1] - 1)
            pos_row = max(ker_idx_row * stride + (local_off // kernel_size) - padding, 0)
            pos_col = max(ker_idx_col * stride + (local_off % kernel_size) - padding, 0)
            
            conv2d_index[i] = (pre_cumsum_origin_patch_flat[i] + 
                            pos_row * height_width_flat[i,1] + 
                            pos_col)
        return conv2d_index
    
    naive_conv2d_index = get_conv2d_index_cpp
    
    conv2d_index = naive_conv2d_index(
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
        kernel_size=kernel_size,
        stride=stride,
    )
    conv2d_index_ref = get_conv2d_index(
        grid_thw=grid_thw,
        spatial_merge_size=spatial_merge_size,
        kernel_size=kernel_size,
        stride=stride,
    )
    
    print(f"{conv2d_index_ref = }")
    print(f"{conv2d_index = }")
    
    assert torch.equal(conv2d_index, conv2d_index_ref), "Results do not match!"
    
    import time
    import random
    
    num_tests = 100
    naive_times = []
    opt_times = []
    
    for i in range(num_tests):
        # 随机生成 grid_thw，保证 h 和 w 都能整除 spatial_merge_size
        num_configs = random.randint(3, 10)
        grid_thw_list = []
        for _ in range(num_configs):
            h = random.randint(10, 100) * spatial_merge_size
            w = random.randint(10, 100) * spatial_merge_size
            t = random.randint(1, 5)
            grid_thw_list.append([t, h, w])
        
        grid_thw = torch.tensor(grid_thw_list, dtype=torch.int32)
        
        print(f"--- Test #{i+1} with grid_thw: {grid_thw_list} total grids: {grid_thw.prod(dim=-1).sum().item()} ---", flush=True)
        
        
        # 测试优化实现
        start = time.time()
        conv2d_index_opt = get_conv2d_index(
            grid_thw=grid_thw,
            spatial_merge_size=spatial_merge_size,
            kernel_size=kernel_size,
            stride=stride,
        )
        opt_time = (time.time() - start) * 1000  # ms
        opt_times.append(opt_time)
        
        print(f"Optimized implementation time: {opt_time:.4f} ms", flush=True)
        
        # 每 10 步对比一次结果
        if (i + 1) % 10 == 0:
            # 测试 naive 实现
            start = time.time()
            conv2d_index_naive = naive_conv2d_index(
                grid_thw=grid_thw,
                spatial_merge_size=spatial_merge_size,
                kernel_size=kernel_size,
                stride=stride,
            )
            naive_time = (time.time() - start) * 1000  # ms
            naive_times.append(naive_time)
            if torch.equal(conv2d_index_naive, conv2d_index_opt):
                print(f"✅ Step {i+1}: Results match! Naive: {naive_time:.4f}ms, Opt: {opt_time:.4f}ms")
            else:
                print(f"❌ Step {i+1}: Results do NOT match!")
                print(f"  grid_thw_list: {grid_thw_list}")
                diff_indices = (conv2d_index_naive != conv2d_index_opt).nonzero(as_tuple=True)[0]
                print(f"  First 10 diff indices: {diff_indices[:10].tolist()}")
                break
    
    # 最终统计
    avg_naive = sum(naive_times) / len(naive_times)
    avg_opt = sum(opt_times) / len(opt_times)
    speedup = avg_naive / avg_opt if avg_opt > 0 else float('inf')
    
    print("\n" + "="*50)
    print("📊 BENCHMARKING SUMMARY")
    print("="*50)
    print(f"Number of tests: {num_tests}")
    print(f"Average Naive Time: {avg_naive:.4f} ms")
    print(f"Average Optimized Time: {avg_opt:.4f} ms")
    print(f"🚀 Speedup Factor: {speedup:.2f}x")
    print("="*50)

'''

P P P P
P P P P
P P 0 1
P P 2 3

-> [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 3]
'''