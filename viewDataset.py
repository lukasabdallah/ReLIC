import torch
import torchvision
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

def  main():
    imagenet_data = torchvision.datasets.ImageNet('/data/public/imagenet2012', split = 'val', transform=transforms.ToTensor())
    # data_loader = torch.utils.data.DataLoader(imagenet_data,
    #                                           batch_size=4,
    #                                           shuffle=True,
    #                                           num_workers=8,
    #                                           pin_memory=True)
    print ("Dataset loaded")


    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(imagenet_data), size=(1,)).item()
        img, label_idx = imagenet_data[sample_idx]
        print(img.size())
        figure.add_subplot(rows, cols, i)
        plt.title(imagenet_data.classes[label_idx])
        plt.axis("off")
        plt.imshow(torch.movedim(img,0,2), )
    plt.show()

if __name__ == "__main__":
    main()