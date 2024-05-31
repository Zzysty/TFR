"""
    训练部分相关函数
"""
import torch


def run_train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, best_model_path):
    print(model)
    print('#---------------------------------------------------------------------------------------------------#')
    print('#--------------------------------------------开始训练模型----------------------------------------------#')
    best_val_loss = float('inf')  # 初始化最低验证损失为无穷大

    for epoch in range(num_epochs):
        model.train()  # 将模型设置为训练模式
        total_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # 清零梯度
            predictions = model(features)  # 前向传播
            loss = criterion(predictions, labels)  # 计算损失
            total_loss += loss.item()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

        # 验证集上的损失计算
        val_loss = 0.0
        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                predictions = model(features)
                val_loss += criterion(predictions, labels).item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}')

        # 如果当前验证损失比最佳损失还低，就保存当前模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)  # 保存模型权重到文件
            print(f"[INFO] Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}).  Saving model ...")

    return model