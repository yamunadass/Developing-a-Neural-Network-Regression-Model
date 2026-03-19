# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features.

In many real-world applications—such as house price prediction, temperature forecasting, sales estimation, or demand prediction—the relationship between input variables and the output is non-linear and complex. Traditional statistical models often fail to capture these patterns effectively. Deep Learning models, particularly Artificial Neural Networks (ANNs), are capable of learning such complex relationships through multiple hidden layers and non-linear activation functions.

In this experiment, a dataset containing multiple independent variables (features) and a dependent variable (target) is provided. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand how network architecture, learning rate, number of epochs, and activation functions affect the accuracy of regression predictions and to demonstrate the effectiveness of deep learning in solving regression problems.

## Neural Network Model
Include the neural network model diagram.
<img width="1095" height="745" alt="542714748-b2141eb2-ad6c-4f3d-8419-c8126c9732b7" src="https://github.com/user-attachments/assets/a10c57a4-bf02-4043-9a23-10da995e3d1c" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Yamuna M

### Register Number: 212223230248

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(ai_brain.parameters(),lr=0.001)




def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
      optimizer.zero_grad()
      Loss=criterion(ai_brain(X_train),y_train)
      Loss.backward()
      optimizer.step()
      ai_brain.history['loss'].append(Loss.item())
      if epoch % 200 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {Loss.item():.6f}')


```

### Dataset Information
<img width="501" height="637" alt="Screenshot 2026-01-30 144014" src="https://github.com/user-attachments/assets/b9a569c2-52f8-4d9e-8af8-12aa9116db73" />


### OUTPUT
<img width="320" height="227" alt="image" src="https://github.com/user-attachments/assets/ae149b4b-afa2-48e6-8323-e032f9d89320" />

<img width="194" height="35" alt="image" src="https://github.com/user-attachments/assets/5f8114f4-d512-4e32-ab2f-2f2560784358" />

### Training Loss Vs Iteration Plot
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/63621001-b232-4a71-9757-e1f5d43fefa5" />


### New Sample Data Prediction
<img width="290" height="35" alt="image" src="https://github.com/user-attachments/assets/fb879bf6-874d-4726-bdc5-73f197a76e65" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
