from archs import LRL, MC_UNet

def split_model(model,model_name=' '):
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        # print("该层的结构：" + str(list(i.size())))
        for j in i.size():
            l *= j
        # print("该层参数和：" + str(l))
        k = k + l
    print(model_name)
    print("总参数数量和：" + str(k))
    print("总层数："+ str(len(params)))
    print("--------------")

split_model(LRL(num_classes=2),'LRL')
split_model(MC_UNet(num_classes=2),'MC_UNet')