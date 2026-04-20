import torch
from torch_geometric.loader import DataLoader

class ModelExecutor():
    def __init__(self, model):
        self.model = model

    def train_model(self, train_data):
        optimizer = torch.optim.Adam(self.model.model.parameters(), lr=0.01, weight_decay=5e-4)
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.model.train()
        for epoch in range(self.epochs+1):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = self.model.model(batch)
                loss = criterion(out,batch.y.long())

                #back prop
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch + 1 == self.epochs:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")


    def eval_model(self, test_data):
        self.model.model.eval()
        loader = DataLoader(test_data, batch_size=self.batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                pred = self.model.model(batch).argmax(dim=1)
                all_preds.append(pred)
                all_labels.append(batch.y.long())
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
        return accuracy 