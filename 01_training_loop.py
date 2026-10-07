import torch

# Data: inputs and the correct answers
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])       # y = 2x + 1

# The model: a real layer with 1 input and 1 output. Inside it: one weight and one bias.
model = torch.nn.Linear(1, 1)
print("weight shape:", model.weight.shape, " bias shape:", model.bias.shape)
print("parameters:  ", list(model.parameters()))

opt = torch.optim.SGD(model.parameters(), lr=0.01)   # optimizer: owns the update rule for these dials

for step in range(200):
    pred = model(x)                     # forward pass: x @ weight.T + bias
    loss = ((pred - y) ** 2).mean()     # how wrong are we? (mean squared error)
    loss.backward()                     # backward pass: fills .grad on weight and bias
    opt.step()                          # update: step opposite the gradient
    opt.zero_grad()                     # clear gradients, or they accumulate
    if step % 40 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}  w {model.weight.item():.3f}  b {model.bias.item():.3f}")
