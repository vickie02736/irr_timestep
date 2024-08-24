import torch
import torch.nn.functional as F
import multiprocessing as mp

def linear_interpolation_worker(batch_frames, known_mask_idx, start, end):
    interpolated_frames = batch_frames.clone()

    for b in range(start, end):
        frames = batch_frames[b]
        known_indices = [
            i for i in range(frames.size(0)) if i not in known_mask_idx[b]
        ]
        missing_indices = known_mask_idx[b].tolist()

        for idx in missing_indices:
            prev_idx = max([i for i in known_indices if i < idx], default=None)
            next_idx = min([i for i in known_indices if i > idx], default=None)

            if prev_idx is None:
                interpolated_frames[b, idx] = frames[next_idx]
            elif next_idx is None:
                interpolated_frames[b, idx] = frames[prev_idx]
            else:
                alpha = (idx - prev_idx) / (next_idx - prev_idx)
                interpolated_frames[b, idx] = (1 - alpha) * frames[prev_idx] + alpha * frames[next_idx]

    return interpolated_frames[start:end]



def linear_interpolation(batch_frames, known_mask_idx, num_workers=8):
    batch_size = batch_frames.size(0)
    chunk_size = (batch_size + num_workers - 1) // num_workers  # Ceiling division

    pool = mp.Pool(num_workers)
    results = []

    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, batch_size)
        if start < end:
            result = pool.apply_async(linear_interpolation_worker, args=(batch_frames, known_mask_idx, start, end))
            results.append(result)

    pool.close()
    pool.join()

    interpolated_frames = torch.cat([result.get() for result in results], dim=0)
    return interpolated_frames



def gaussian_interpolation_worker(batch_frames, known_mask_idx, kernel, kernel_size, start, end, is_spatial):
    interpolated_frames = batch_frames.clone()

    for b in range(start, end):
        if is_spatial:
            batch_size, seq_len, channels, height, width = batch_frames.shape
            for c in range(channels):
                for h in range(height):
                    for w in range(width):
                        values = batch_frames[b, :, c, h, w]
                        known_mask = torch.ones(seq_len, device=batch_frames.device)
                        known_mask[known_mask_idx[b]] = 0

                        smoothed_values = F.conv1d(values.view(1, 1, -1) * known_mask.view(1, 1, -1),
                                                   kernel, padding=kernel_size // 2)
                        smoothed_known = F.conv1d(known_mask.view(1, 1, -1),
                                                  kernel, padding=kernel_size // 2)

                        smoothed_values /= smoothed_known + 1e-10
                        interpolated_frames[b, :, c, h, w] = smoothed_values.view(-1)
        else:
            batch_size, seq_len, latent_dim = batch_frames.shape
            for l in range(latent_dim):
                values = batch_frames[b, :, l]
                known_mask = torch.ones(seq_len, device=batch_frames.device)
                known_mask[known_mask_idx[b]] = 0

                smoothed_values = F.conv1d(values.view(1, 1, -1) * known_mask.view(1, 1, -1),
                                           kernel, padding=kernel_size // 2)
                smoothed_known = F.conv1d(known_mask.view(1, 1, -1),
                                          kernel, padding=kernel_size // 2)

                smoothed_values /= smoothed_known + 1e-10
                interpolated_frames[b, :, l] = smoothed_values.view(-1)

    return interpolated_frames[start:end]


def gaussian_interpolation(batch_frames, known_mask_idx, sigma=1.0, num_workers=8):
    if batch_frames.dim() == 5:
        batch_size, seq_len, channels, height, width = batch_frames.shape
        is_spatial = True
    elif batch_frames.dim() == 3:
        batch_size, seq_len, latent_dim = batch_frames.shape
        is_spatial = False
    else:
        raise ValueError("Input batch_frames must be a 3D or 5D tensor")

    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = gaussian_kernel1d(kernel_size, sigma).view(1, 1, -1).to(batch_frames.device)

    chunk_size = (batch_size + num_workers - 1) // num_workers

    pool = mp.Pool(num_workers)
    results = []

    for i in range(num_workers):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, batch_size)
        if start < end:
            result = pool.apply_async(gaussian_interpolation_worker, args=(batch_frames, known_mask_idx, kernel, kernel_size, start, end, is_spatial))
            results.append(result)

    pool.close()
    pool.join()

    interpolated_frames = torch.cat([result.get() for result in results], dim=0)
    return interpolated_frames

# def linear_interpolation(batch_frames, known_mask_idx):

#     if batch_frames.dim() == 5:
#         batch_size, seq_len, channels, height, width = batch_frames.shape
#         is_spatial = True
#     elif batch_frames.dim() == 3:
#         batch_size, seq_len, latent_dim = batch_frames.shape
#         is_spatial = False
#     else:
#         raise ValueError("Input batch_frames must be 3D or 5D tensor")

#     interpolated_frames = batch_frames.clone()

#     for b in range(batch_size):
#         frames = batch_frames[b]
#         known_indices = [
#             i for i in range(seq_len) if i not in known_mask_idx[b]
#         ]
#         missing_indices = known_mask_idx[b].tolist()

#         for idx in missing_indices:
#             prev_idx = max([i for i in known_indices if i < idx], default=None)
#             next_idx = min([i for i in known_indices if i > idx], default=None)

#             if prev_idx is None:
#                 interpolated_frames[b, idx] = frames[next_idx]
#             elif next_idx is None:
#                 interpolated_frames[b, idx] = frames[prev_idx]
#             else:
#                 alpha = (idx - prev_idx) / (next_idx - prev_idx)
#                 interpolated_frames[b, idx] = (
#                     1 - alpha) * frames[prev_idx] + alpha * frames[next_idx]

#     return interpolated_frames


def gaussian_kernel1d(size, sigma):
    coords = torch.arange(size).float() - (size - 1) / 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    g /= g.sum()
    return g

# def gaussian_interpolation(batch_frames, known_mask_idx, sigma=1.0):
#     if batch_frames.dim() == 5:
#         batch_size, seq_len, channels, height, width = batch_frames.shape
#         is_spatial = True
#     elif batch_frames.dim() == 3:
#         batch_size, seq_len, latent_dim = batch_frames.shape
#         is_spatial = False
#     else:
#         raise ValueError("Input batch_frames must be a 3D or 5D tensor")

#     interpolated_frames = batch_frames.clone()

#     kernel_size = int(6 * sigma + 1)
#     if kernel_size % 2 == 0:
#         kernel_size += 1

#     kernel = gaussian_kernel1d(kernel_size, sigma).view(1, 1, -1).to(batch_frames.device)

#     for b in range(batch_size):
#         if is_spatial:
#             for c in range(channels):
#                 for h in range(height):
#                     for w in range(width):
#                         values = batch_frames[b, :, c, h, w]
#                         known_mask = torch.ones(seq_len, device=batch_frames.device)
#                         known_mask[known_mask_idx[b]] = 0
#                         smoothed_values = F.conv1d(values.view(1, 1, -1) * known_mask.view(1, 1, -1),
#                                                    kernel, padding=kernel_size // 2)
#                         smoothed_known = F.conv1d(known_mask.view(1, 1, -1),
#                                                   kernel, padding=kernel_size // 2)

#                         smoothed_values /= smoothed_known + 1e-10  # Avoid division by zero
#                         interpolated_frames[b, :, c, h, w] = smoothed_values.view(-1)
#         else:
#             for l in range(latent_dim):
#                 values = batch_frames[b, :, l]
#                 known_mask = torch.ones(seq_len, device=batch_frames.device)
#                 known_mask[known_mask_idx[b]] = 0

#                 smoothed_values = F.conv1d(values.view(1, 1, -1) * known_mask.view(1, 1, -1),
#                                            kernel, padding=kernel_size // 2)
#                 smoothed_known = F.conv1d(known_mask.view(1, 1, -1),
#                                           kernel, padding=kernel_size // 2)

#                 smoothed_values /= smoothed_known + 1e-10  # Avoid division by zero
#                 interpolated_frames[b, :, l] = smoothed_values.view(-1)

#     return interpolated_frames


# Example usage
# x = torch.rand(2, 10, 3, 128, 128)
# from tools import mask
# x, idx = mask(x)
# interpolated_frames = linear_interpolation(x, idx)
# # interpolated_frames = gaussian_interpolation(x, idx)
# print(interpolated_frames.shape)
