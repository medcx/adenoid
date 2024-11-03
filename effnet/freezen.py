
def freeze(model,device):
    model.to(device)
    for param in model.model.parameters():
        param.requires_grad = False

    # 解冻最后几个卷积块
    for param in model.model._conv_head.parameters():
        param.requires_grad = True
    for param in model.model._bn1.parameters():
        param.requires_grad = True

    # 解冻全连接层
    for param in model.model._fc.parameters():
        param.requires_grad = True

    return model
