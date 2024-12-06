from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN


def train():
    transform = v2.Compose(
        [
            v2.Resize((356, 356)),
            v2.RandomCrop((299, 299)),
            # v2.ToTensor(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale = True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
     
    train_loader, dataset = get_loader(
        root_folder = "C:\\Users\\Srijan\\Desktop\\Srijan\\seq2seq-demo\\image_captioning\\Flickr_8k_Images_Captions\\flickr8k\\images\\",
        annotation_file = "C:\\Users\\Srijan\\Desktop\\Srijan\\seq2seq-demo\\image_captioning\\Flickr_8k_Images_Captions\\flickr8k\\captions.txt",
        transform=transform,
        num_workers = 6
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("runs/cnn2lstm")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)
    
    model.train()

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch+1} / {num_epochs}]")
        print_examples(model, device, dataset)
        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }
            save_checkpoint(checkpoint)
        
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}'):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            # outputs: (seq_len, N, vocab_size); targets: (seq_len, N)
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    train()
