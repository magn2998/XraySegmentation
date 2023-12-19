import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.transforms.functional import crop
import torch
import torch.nn as nn
from collections import defaultdict
import cv2
import Unet_old

def Use_Model_On_Specific_Image_Using_Sections(imageNo, trainedSize, modelPath, saveFig, useCpu) :


    predictionLocation = "./images/segmentation_Prediction.png"

    imagePath = "./data/data/SOCprist0" + imageNo + ".tiff"
    imageLabel = "slice__" + imageNo + ".tif"


    SEGMENTS_WIDTH   = trainedSize # Height of individual segments, which are cropped section of the original image
    SEGMENTS_HEIGHT  = trainedSize # Same as above, just for height
    SEGMENTS_OVERLAP = 10  # Pixels to overlap between segments


    # Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device Used: " + str(device))


    # Crop Image
    def crop(img, start_x, start_y, width, height):
        return img[start_y:start_y+height, start_x:start_x+width]

    # Merge segment sections back together
    def merge_segment(full_img, segment, start_x, start_y, end_x, end_y, segment_x, segment_y):
        temp_start_y = start_y
        temp_segment_y = segment_y
        while start_x < end_x:
            while start_y < end_y:
                full_img[0][start_y][start_x] = segment[0][segment_y][segment_x]
                full_img[1][start_y][start_x] = segment[1][segment_y][segment_x]
                full_img[2][start_y][start_x] = segment[2][segment_y][segment_x]

                start_y += 1
                segment_y += 1
            start_y = temp_start_y
            segment_y = temp_segment_y
            start_x += 1 
            segment_x += 1   

    # Merge all segments together to a single image
    def merge_all_segments(img_arr, segments, seg_size, min_overlap):
        H, W = img_arr.shape
        Ws, Hs = seg_size

        # Create Full Array
        full_arr = np.zeros((3, W, H))

        # Merge the segments
        idx_x = 0
        i = 0
        while i < W:
            o = 0
            idx_y = 0
            while o < H:
                start_x = int(max(i - min_overlap/2, 0))
                start_y = int(max(o - min_overlap/2, 0))

                end_x = int(min(max(i - min_overlap, 0)+Ws-min_overlap/2, W))
                end_y = int(min(max(o - min_overlap, 0)+Hs-min_overlap/2, H))

                # Edge case - If last segment just goes to edge, just take entire thing
                if max(i - min_overlap, 0)+Ws == W:
                    end_x = W
                if max(o - min_overlap, 0)+Hs == H:
                    end_y = H

                # Now that we've computed the placements inside the full picture, compute the indexes from the segments
                segment_x = max(i - min_overlap, 0)
                segment_y = max(o - min_overlap, 0)

                # If segments overflows, make segment so it covers rest while maintaining size
                if segment_x + Ws >= W:
                    segment_x = W - Ws
                if segment_y + Hs >= H:
                    segment_y = H - Hs

                segment_x = start_x - segment_x
                segment_y = start_y - segment_y

                merge_segment(full_arr, segments[idx_x][idx_y], start_x, start_y, end_x, end_y, segment_x, segment_y)
                
                idx_y += 1
                o = max(o - min_overlap, 0) + Hs

            idx_x += 1
            i = max(i - min_overlap, 0) + Ws

        return torch.from_numpy(full_arr)



    # Create Image Segmentations
    def create_segments(img_arr, seg_size, min_overlap):
        H, W = img_arr.shape
        Ws, Hs = seg_size
        O = min_overlap

        # Segment the array
        segments = []

        idx_x = 0
        i = 0
        while i < W:
            o = 0
            idx_y = 0
            segments.append([])

            while o < H:
                segments[idx_x].append([np.zeros((Ws, Hs))])

                segment_x = max(i - min_overlap, 0)
                segment_y = max(o - min_overlap, 0)

                # If segments overflows, make segment so it covers rest while maintaining size
                if segment_x + Ws >= W:
                    segment_x = W - Ws
                if segment_y + Hs >= H:
                    segment_y = H - Hs

                segments[idx_x][idx_y] = crop(img_arr, segment_x, segment_y, Ws, Hs)
                
                
                idx_y += 1
                o = segment_y + Hs

            
            idx_x += 1
            i = segment_x + Ws

        return segments

    def dice_loss(pred, target):
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        total_sum = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2);
        
        dice = 2.0*intersection/total_sum

        return (1 - dice).mean()

    def calc_loss(pred, target, metrics, criterion, bce_weight=0.5):
        cce = criterion(pred, target)

        SoftMaxFunc = nn.Softmax2d()
        pred = SoftMaxFunc(pred)

        dice = dice_loss(pred, target)
        # IoU = IoU_loss(pred, target)
        # print(IoU)

        loss = cce * bce_weight + dice * (1 - bce_weight)

        metrics['cce'] += cce.data.cpu().numpy() * target.size(0)
        metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
        metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss


    def print_metrics(metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))


    # Prepare Image
    transformers = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    image = cv2.imread(imagePath, 0) # Read as grayscaled 
    image = transformers(image)
    image = torch.squeeze(image)
    print(image.shape)

    # Setup Model
    model = Unet_old.UNet(3).to(device)
    if useCpu:
        model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(modelPath))
    model.eval() 

    # Split image into smaller overlapping images
    segments = create_segments(image, (SEGMENTS_WIDTH, SEGMENTS_HEIGHT), SEGMENTS_OVERLAP)

    # Make Prediction
    for i in range(len(segments)):
        for o in range(len(segments[i])):
            print("Analysing Segment: " + str(i*len(segments)+o))
            input = segments[i][o].unsqueeze(0).unsqueeze(0).to(device)
            pred = model(input)
            pred = pred.data.cpu().numpy()
            segments[i][o] = pred.squeeze(0)

    # Combine Images
    merged = merge_all_segments(image, segments, (SEGMENTS_WIDTH, SEGMENTS_HEIGHT), SEGMENTS_OVERLAP)

    # Read True Label
    blackmask = cv2.imread("./data/split_masks/black/" + imageLabel, 0) 
    greymask  = cv2.imread("./data/split_masks/grey/"  + imageLabel, 0) 
    whitemask = cv2.imread("./data/split_masks/white/" + imageLabel, 0) 

    blackmask = transformers(blackmask)
    greymask  = transformers(greymask)
    whitemask = transformers(whitemask)

    total_mask = torch.zeros((3, blackmask.size(1), blackmask.size(2)))
    total_mask[0] = blackmask
    total_mask[1] = greymask
    total_mask[2] = whitemask

    # Unsqueeze
    merged = merged.unsqueeze(0)
    total_mask = total_mask.unsqueeze(0)

    # Calculate Loss
    metrics = defaultdict(float)
    criterion = nn.CrossEntropyLoss()
    loss = calc_loss(merged, total_mask, metrics, criterion)
    print(loss)
    print_metrics(metrics, 1, "val")

    # Save Result
    img_to_plot = merged.softmax(dim=1)[0].permute(1, 2, 0)
    plt.figure(figsize=(8, 8))
    plt.tight_layout()
    plt.imshow(img_to_plot, cmap="gray")
    if saveFig:
        plt.savefig(predictionLocation)