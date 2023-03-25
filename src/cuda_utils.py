import torch

def get_cuda_details():
    """
    This function returns a dictionary containing the details of the available CUDA devices.

    The returned dictionary contains the following keys:

        "torch.cuda.is_available()": A boolean value indicating whether CUDA is available or not.
        "torch.cuda.device_count()": An integer value indicating the number of CUDA devices.
        "torch.cuda.device(current_device)": A string representation of the current CUDA device.
        "torch.cuda.get_device_name(current_device)": A string representation of the name of the current CUDA device.

    :param None
    :return: A dictionary containing the details of the available CUDA devices.
    """
    cuda_avaliable = torch.cuda.is_available()
    if cuda_avaliable:
        return {
            "torch.cuda.is_available()": cuda_avaliable,
            "torch.cuda.device_count()": torch.cuda.device_count(),
            "torch.cuda.device(current_device)": torch.cuda.device(torch.cuda.current_device()),
            "torch.cuda.get_device_name(current_device)": torch.cuda.get_device_name(0)}
    else:
        return {"torch.cuda.is_available()": cuda_avaliable}