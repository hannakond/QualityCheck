import torch

def get_cuda_details():
    cuda_avaliable = torch.cuda.is_available()
    if cuda_avaliable:
        return {
            "torch.cuda.is_available()": cuda_avaliable,
            "torch.cuda.device_count()": torch.cuda.device_count(),
            "torch.cuda.device(current_device)": torch.cuda.device(torch.cuda.current_device()),
            "torch.cuda.get_device_name(current_device)": torch.cuda.get_device_name(0)}
    else:
        return {"torch.cuda.is_available()": cuda_avaliable}