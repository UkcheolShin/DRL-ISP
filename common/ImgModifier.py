import numpy as np
import cv2
import random
import scipy.ndimage as ndi

def show_img_cv2_plt(image) :
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

class ImgModifier(object):
    def __init__(self, level='basic'):
        super(ImgModifier, self).__init__()
        self.seed  = 3598
        random.seed(self.seed)
        self.num_blur = 3
        self.num_addnoise = 1
        self.num_brightness = 3

        # noise level
        self.noise_basic_lv = [0, 5.0]
        self.noise_low_lv = [3.0, 25.0]
        self.noise_high_lv = [25.0, 50.0]
        self.noise_all_lv = [3.0, 50.0]

        # blur kernel size
        self.blur_basic_lv = [[3], [0.15, 1.125], [3,5]]
        self.blur_low_lv = [[3,5,7], [0.15, 2.25], [3,5,7,9,11]] 
        self.blur_high_lv = [[7,9,11], [2.25, 4.5], [11,13,15,17]]
        self.blur_all_lv = [[3,5,7,9,11], [0.15, 4.5], [3,5,7,9,11,13,15,17]]

        # x2, x3
        self.sr_basic_lv = [1,2]
        self.sr_low_lv = [2]
        self.sr_high_lv = [3]
        self.sr_all_lv = [2,3]

        # Compression ratio
        self.jpeg_basic_lv = [60,95]
        self.jpeg_low_lv = [25,90]
        self.jpeg_high_lv = [10,35]
        self.jpeg_all_lv = [10,100]

        # brightness level
        self.bright_basic_lv = [[-0.25, 0.25], [0.5, 1.5], [0.8, 1.5]]
        self.bright_low_lv = [[-0.5, 0.5], [0.5, 1.5], [0.4, 2.0]]
        self.bright_high_lv = [[-1.0, 1.0], [0.2, 2.0], [0.2, 3.5]]
        self.bright_all_lv = [[-1.0, 1.0], [0.2, 2.0], [0.2, 3.5]]
        self.lv = level
        self.set_modifier_lv(self.lv)

    def set_modifier_lv(self, lv) :
        if lv == 'basic' : 
            self.noise_lv = self.noise_basic_lv
            self.blur_lv = self.blur_basic_lv
            self.sr_lv = self.sr_basic_lv
            self.jpeg_lv = self.jpeg_basic_lv
            self.bright_lv = self.bright_basic_lv
            print('Set to Modifer level : BASIC')
        elif lv == 'low':
            self.noise_lv = self.noise_low_lv
            self.blur_lv = self.blur_low_lv
            self.sr_lv = self.sr_low_lv
            self.jpeg_lv = self.jpeg_low_lv
            self.bright_lv = self.bright_low_lv
            print('Set to Modifer level : LOW')
        elif lv == 'high':
            self.noise_lv = self.noise_high_lv
            self.blur_lv = self.blur_high_lv
            self.sr_lv = self.sr_high_lv
            self.jpeg_lv = self.jpeg_high_lv
            self.bright_lv = self.bright_high_lv
            print('Set to Modifer level : HIGH')
        elif lv == 'all':
            self.noise_lv = self.noise_all_lv
            self.blur_lv = self.blur_all_lv
            self.sr_lv = self.sr_all_lv
            self.jpeg_lv = self.jpeg_all_lv
            self.bright_lv = self.bright_all_lv
            print('Set to Modifer level : ALL')

    # Data augmentation
    def add_basic_effect(self, img, Itype) :
        assert type(img) == np.ndarray # HWC
        assert self.lv == 'basic' # HWC

        # lr -> blur -> brightness -> noise -> jpeg
        if Itype == 'noise' :     
            outimg  = self.addblur(img)
        elif Itype == 'blur' :
            outimg  = self.addnoise(img)
        elif Itype == 'sr' :
            outimg  = self.addblur(img)
            outimg  = self.addnoise(outimg)
        elif Itype == 'jpeg' :
            outimg  = self.addblur(img)
            outimg  = self.addnoise(outimg)
        elif Itype == 'brightness' :
            outimg  = self.addblur(img)
            outimg  = self.addnoise(outimg)
        elif Itype == 'raw' :
            outimg  = self.addblur(img)
            outimg  = self.addnoise(outimg)
        else :
            outimg  = self.addblur(img)
            outimg  = self.addnoise(outimg)
        return outimg

    def make_lowresol(self, img, scale=255, view_opt = False) :
        hr_height, hr_width, _  = img.shape
        if scale == 255:
            if len(self.sr_lv) > 1 :
                scale = random.randint(self.sr_lv[0],self.sr_lv[1])
            else:
                scale = self.sr_lv[0]
        lr = cv2.resize(img, (hr_width // scale, hr_height // scale), interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(lr, (hr_width, hr_height), interpolation=cv2.INTER_CUBIC)
        if view_opt : 
            return lr.astype(np.uint8), scale, scale
        else :
            return lr.astype(np.uint8)

    def addblur(self, img, Ftype=255, view_opt = False):
        assert type(img) == np.ndarray # HWC
        # choose blur function
        if Ftype == 255:
            Ftype = random.randint(0, self.num_blur-1)

        # decide blur level
        if Ftype == 0: # boxfilter 
            kernel_size = random.choice(self.blur_lv[0])
            outimg = cv2.blur(img, ksize=(kernel_size, kernel_size))
        elif Ftype == 1: # gaussian filter
            sigma = random.uniform(self.blur_lv[1][0],self.blur_lv[1][1])
            kernel_size = 2*int(4*sigma + 0.5) + 1
            outimg = cv2.GaussianBlur(img, ksize=(kernel_size,kernel_size), sigmaX=sigma)
        elif Ftype == 2: # motion blur, ref. kornia. get_motion_kernel2d
            kernel_size = random.choice(self.blur_lv[2])
            angle = random.randint(0, 359)
            direction = 0.5 #random.uniform(0.,1.) # 1 : forward, 0.5: uniform, 0 : backward
            motion_blur = np.zeros((kernel_size, kernel_size))
            direct = [direction + (1-2*direction) / (kernel_size - 1) * i for i in range(kernel_size)]
            motion_blur[int((kernel_size-1)/2), :] = np.array(direct)
            motion_blur = ndi.rotate(motion_blur, angle, reshape=False)
            motion_blur = motion_blur / motion_blur.sum()
            outimg = cv2.filter2D(img, -1, motion_blur)

        if view_opt : 
            return outimg.astype(np.uint8), Ftype, kernel_size
        else :
            return outimg.astype(np.uint8)

    def addnoise(self, img, Ftype=255, view_opt = False):
        assert type(img) == np.ndarray # HWC
        # choose function
        # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        if Ftype == 255:
            Ftype = random.randint(0, self.num_addnoise-1)

        if Ftype == 0: # Gaussian noise
            mean = 0
            std = random.uniform(self.noise_lv[0],self.noise_lv[1])
            noise = np.random.normal(mean, std, (img.shape)) # gaussian distribution
            outimg = img + noise
            prob = std # for the debug
        # elif Ftype == 1: # Salt & Papper
        #     prob = random.uniform(self.noise_lv[0],self.noise_lv[1])/255*3 # convert to probability, about range : 0~0.6
        #     outimg = np.copy(img)
        #     rdn_array = np.random.random(img.shape)
        #     salt_mask = rdn_array < prob
        #     outimg[salt_mask] = 1
        #     pepper_mask = rdn_array > 1 - prob
        #     outimg[pepper_mask] = 255
        # elif Ftype == 2: # bernoulli
        #     """
        #     https://github.com/shivamsaboo17/Deep-Restore-PyTorch/blob/master/data.py
        #     Multiplicative bernoulli
        #     """
        #     prob = random.uniform(self.noise_lv[0],self.noise_lv[1])/255*3 # convert to probability
        #     mask = np.random.choice([0, 1], size=np.array(img).shape, p=[prob, 1 - prob])
        #     outimg = np.multiply(img, mask).astype(np.uint8)
        # elif Ftype == 3: # poisson
        #     vals = len(np.unique(img))
        #     vals = 2 ** np.ceil(np.log2(vals))
        #     outimg = np.random.poisson(img * vals) / float(vals)

        if view_opt : 
            return outimg.clip(0,255).astype(np.uint8), Ftype, prob
        else :
            return outimg.clip(0,255).astype(np.uint8)

    def jpg_comp(self, img, Ftype=255, view_opt = False):
        assert type(img) == np.ndarray # HWC
        jpeg_quality = random.randint(self.jpeg_lv[0],self.jpeg_lv[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

        outimg = np.zeros(img.shape)
        for idx, im in zip(range(len(img.transpose(2,0,1))) ,img.transpose(2,0,1)) : 
            result, encimg = cv2.imencode('.jpg', im, encode_param) # jpeg encode
            outim = cv2.imdecode(encimg,cv2.IMREAD_UNCHANGED)                          # jpeg decode
            outimg[:,:,idx]= outim

        if view_opt : 
            return outimg.astype(np.uint8), Ftype, jpeg_quality
        else :
            return outimg.astype(np.uint8)

    def change_brightness(self, img, Ftype=255, view_opt = False):
        assert type(img) == np.ndarray

        if Ftype == 255 :  
            Ftype = random.randint(0, self.num_brightness-1)

        if Ftype == 0 :  # simple addition
            factor = random.uniform(self.bright_lv[0][0],self.bright_lv[0][1])               
            outimg = (img + factor*128.).clip(0,255).astype(np.uint8) # -64~64, -128~128
        elif Ftype == 1: # simple multiplication (contrast adjustment)
            factor = random.uniform(self.bright_lv[1][0],self.bright_lv[1][1])               
            outimg = (img * factor).clip(0,255).astype(np.uint8) # 0.2 ~ 2.0, 
        elif Ftype == 2: # gamma correction
            # https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
            # correct brightness with non linear transformation
            #outimg = gain * img^factor
            gamma = random.uniform(self.bright_lv[2][0],self.bright_lv[2][1])               
            lookUpTable = np.empty((1,256), np.uint8)
            for i in range(256):        
                lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

            outimg = np.zeros(img.shape)
            for idx, im in zip(range(len(img.transpose(2,0,1))),img.transpose(2,0,1)) : 
                outim = cv2.LUT(im.astype(np.uint8), lookUpTable).astype(np.uint8)
                outimg[:,:,idx]= outim

            factor = gamma

        if view_opt : 
            return outimg, Ftype, factor
        else :
            return outimg
            
    #@todo implement deblur filter c++ --> python
    #https://docs.opencv.org/master/de/d3c/tutorial_out_of_focus_deblur_filter.html
    #https://docs.opencv.org/4.1.1/d1/dfd/tutorial_motion_deblur_filter.html
    def deblur(self, img): # sharpening
        assert type(img) == np.ndarray # HWC
        Ftype = random.randint(0, self.num_deblur-1)
        if Ftype == 0: # Gaussian noise
            sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]) 
            outimg = cv2.filter2D(img, -1, sharpening_1)

        return outimg.astype(np.uint8)

    #https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
    #https://blog.naver.com/laonple/220834097950
    def denoise(self, img):
        assert type(img) == np.ndarray # HWC
        if Ftype == 255:
            Ftype = random.randint(0, self.num_deblur-1)
        if Ftype == 0: # Gaussian noise
            outimg = cv2.fastNlMeansDenoisingColored(h = 10, hForColorComponents=10, templateWindowSize=7, searchWindowSize=21)
        return outimg

    def contrast_enhance(self, img, Ftype=255):
        assert type(img) == np.ndarray # HWC
        # choose function
        if Ftype == 255:
            Ftype = random.randint(0, self.num_contrast-1)
        img_yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV) # Y : intensity, u,v : color

        if Ftype == 0: # Histogram Equlization
            img_y = cv2.equalizeHist(img_yuv[:,:,0])
        elif Ftype == 1: # CLAHE(Contrast Limited Adaptive Histogram Equalization) 
            params1 = [1,3,5]
            param1 = random.choice(params1)
            params2 = [5,7,9,11]
            param2 = random.choice(params2)

            clahe = cv2.createCLAHE(clipLimit=param1, tileGridSize=(param2,param2))
            img_y = clahe.apply(img_yuv[:,:,0])

        img_yuv[:,:,0] = img_y
        outimg = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2RGB) # Y : intensity, u,v : color
        return outimg

    #https://www.learnopencv.com/high-dynamic-range-hdr-imaging-using-opencv-cpp-python/
    def Multishot_HDR(self, imgs, times):
        # List of exposure times
        times = np.array([ 1/30.0, 0.25, 2.5, 15.0 ], dtype=np.float32)
        # List of image filenames
        filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]
        imgs = []
        for filename in filenames:
            im = cv2.imread(filename)
            imgs.append(im)

        # Align input images
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(imgs, imgs)

        # Obtain Camera Response Function (CRF)
        calibrateDebevec = cv2.createCalibrateDebevec()
        responseDebevec = calibrateDebevec.process(imgs, times)
        # Merge images into an HDR linear image
        mergeDebevec = cv2.createMergeDebevec()
        hdrDebevec = mergeDebevec.process(imgs, times, responseDebevec)

        # Save HDR image.
        cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
        outimg = hdrDebevec
        return outimg

    def tone_mapping(self, hdr_img, Ftype=255) :
        assert type(hdr_img) == np.ndarray # HWC
        # choose function
        if Ftype == 255:
            Ftype = random.randint(0, self.num_tone_mapping-1)
        if Ftype == 0:
            # Tonemap using Drago's method to obtain 24-bit color image
            tonemapDrago = cv2.createTonemapDrago(gamma = 1.0, saturation = 0.7, bias=0.85)
            ldrDrago = tonemapDrago.process(hdr_img)
            ldrDrago = 3 * ldrDrago
            cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
        elif Ftype == 1: 
            # Tonemap using Durand's method obtain 24-bit color image
            tonemapDurand = cv2.createTonemapDurand(gamma = 1.5, contrast = 4, saturation = 1.0, sigma_space = 1, sigma_color = 1)
            ldrDurand = tonemapDurand.process(hdr_img)
            ldrDurand = 3 * ldrDurand
            cv2.imwrite("ldr-Durand.jpg", ldrDurand * 255)
        elif Ftype == 2 :
            # Tonemap using Reinhard's method to obtain 24-bit color image
            tonemapReinhard = cv2.createTonemapReinhard(gamma = 1.5, intensity = 0, light_adapt = 0, color_adapt = 0)
            ldrReinhard = tonemapReinhard.process(hdr_img)
            cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
        elif Ftype == 3 :
            # Tonemap using Mantiuk's method to obtain 24-bit color image
            tonemapMantiuk = cv2.createTonemapMantiuk(gamma = 2.2, scale = 0.85, saturation = 1.2)
            ldrMantiuk = tonemapMantiuk.process(hdr_img)
            ldrMantiuk = 3 * ldrMantiuk
            cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)

        outimg = img
        return outimg

    # https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
    def demosaic(self, img, Ftype=255) :
        assert type(img) == np.ndarray
        outimg = img
        return outimg

    def colorspace_conversion(self, img, Ftype=255):
        assert type(img) == np.ndarray
        outimg = img
        return outimg

    #https://stackoverflow.com/questions/46390779/automatic-white-balancing-with-grayworld-assumption/46391574
    # grayworld assuption, basic white-balcne method.(von kries algorithm)
    def white_balance(self, img):
        assert type(img) == np.ndarray
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        avg_a = np.average(img_lab[:, :, 1])
        avg_b = np.average(img_lab[:, :, 2])
        img_lab[:, :, 1] = img_lab[:, :, 1] - ((avg_a - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
        img_lab[:, :, 2] = img_lab[:, :, 2] - ((avg_b - 128) * (img_lab[:, :, 0] / 255.0) * 1.1)
        outimg = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        return outimg
