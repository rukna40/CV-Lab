import cv2
import numpy as np

def build_gp(img, levels):
    gp = [img]
    for _ in range(levels - 1):
        blurred = cv2.GaussianBlur(gp[-1], (5, 5), 0)
        down = cv2.resize(blurred, (int(gp[-1].shape[1] * 0.5), int(gp[-1].shape[0] * 0.5)), interpolation=cv2.INTER_AREA)
        gp.append(down)
    return gp

def compute_dog(gp):
    dog = []
    for i in range(len(gp) - 1):
        img1 = gp[i]
        img2 = gp[i + 1]
        if img1.shape != img2.shape:
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)
        dog.append(cv2.subtract(img2, img1))
    return dog

def find_kps(dog):
    kps = []
    for i in range(1, len(dog) - 1):
        curr, prev, next = dog[i], dog[i - 1], dog[i + 1]
        for y in range(1, curr.shape[0] - 1):
            for x in range(1, curr.shape[1] - 1):
                if is_ext(x, y, curr, prev, next):
                    kps.append((x, y, i))
    return kps

def is_ext(x, y, curr, prev, next):
    if (curr[y-1:y+2, x-1:x+2].size == 0 or 
        prev[y-1:y+2, x-1:x+2].size == 0 or 
        next[y-1:y+2, x-1:x+2].size == 0):
        return False
    p = curr[y, x]
    return (p == np.max(curr[y-1:y+2, x-1:x+2]) and
            p == np.max(prev[y-1:y+2, x-1:x+2]) and
            p == np.max(next[y-1:y+2, x-1:x+2])) or (
            p == np.min(curr[y-1:y+2, x-1:x+2]) and
            p == np.min(prev[y-1:y+2, x-1:x+2]) and
            p == np.min(next[y-1:y+2, x-1:x+2]))

def compute_desc(img, kps, patch_size=32):
    descs = []
    for x, y, _ in kps:
        x, y = int(x), int(y)
        patch = img[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx**2 + gy**2)
            descs.append(grad.flatten())
    return np.array(descs)

def match_desc(desc1, desc2):
    return [(i, np.argmin(np.linalg.norm(desc2 - d1, axis=1))) for i, d1 in enumerate(desc1)]

def main():
    img = cv2.imread('/home/ankur/Codes/Sem5/CV-Lab/Lab5/house.png', 0)
    gp = build_gp(img, 4)
    dog = compute_dog(gp)
    kps = find_kps(dog)
    descs = compute_desc(img, kps)
    matches = match_desc(descs, descs)
    
    
    img_kps = cv2.drawKeypoints(img, [cv2.KeyPoint(x, y, 1) for x, y, _ in kps], None)
    cv2.imshow('Keypoints', img_kps)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
