def forward(self, X):
    Y = self.bn1(self.conv1(X))
    Y = nd.relu(Y)
    Y = self.bn2(self.conv2(Y))
    if self.con3:
        X = self.conv3(X)
    return nd.relu(Y+X)