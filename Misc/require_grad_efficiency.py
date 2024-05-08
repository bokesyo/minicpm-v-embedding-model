import torch
import torch.nn as nn
import time

# 创建一个简单的Transformer模型

class SimpleTransformerModel(nn.Module):
    def __init__(self):
        super(SimpleTransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=2304, 
            dim_feedforward=5760,
            nhead=36)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=40) # this will not replicate, safe to use.

    def forward(self, src):
        output = self.transformer_encoder(src)
        return output

# 测试函数
def test_model(requires_grad, backward, device):
    model = SimpleTransformerModel().to(device)
    # print(model.state_dict().keys())
    # 配置模型参数的requires_grad属性
    for param in model.parameters():
        param.requires_grad = requires_grad
    
    src = torch.rand(16, 512, 2304, device=device)  # (sequence length, batch size, feature number)
    
    if not requires_grad:        

        start_time = time.time()
        with torch.no_grad():
            output = model(src)
            loss = torch.sum(output)

        end_time = time.time()
    else:

        start_time = time.time()
        output = model(src)
        loss = torch.sum(output)
        if backward:
            loss.backward()
        
        end_time = time.time()
    
    # 获取显存占用
    memory_used = torch.cuda.memory_allocated(device)
    return memory_used, end_time - start_time

# 主程序
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 测试 requires_grad = True
# mem_used_grad, time_grad = test_model(True, False, device)
# print(f"With requires_grad=True: Memory = {mem_used_grad} bytes, Inference Time = {time_grad} seconds")


# # 测试 requires_grad = False
# mem_used_no_grad, time_no_grad = test_model(False, False, device)
# print(f"With requires_grad=False: Memory = {mem_used_no_grad} bytes, Inference Time = {time_no_grad} seconds")



# # 测试 requires_grad = True + backward
mem_used_grad, time_grad = test_model(True, True, device)
print(f"With requires_grad=True & backward(): Memory = {mem_used_grad} bytes, Total Time = {time_grad} seconds")

