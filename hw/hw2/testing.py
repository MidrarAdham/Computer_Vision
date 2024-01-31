import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import minimize


class calibratePinholeCamera():
    
    def __init__(self):
        self.sample_data_dir = f'../../sample_data/'
        
        self.world_3d_points = []
        self.img_plane_2d_points = []
        
        self.image_1 = 'pattern_1.jpg'
        self.image_2 = 'pattern_2.jpg'
        
        squares_size = 25 # mm
        chessboard_len = 6
        chessboard_width = 9
        
        self.object_pts = np.zeros((chessboard_len*chessboard_width,3), np.float32)
        self.object_pts[:,:2] = np.mgrid[0:chessboard_width,0:chessboard_len].T.reshape(-1,2) * squares_size
    
    def read_images (self, img):
        self.img = cv.imread(self.sample_data_dir+img)

    def multi_image_plot(self, img1, img2):
        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,10))
        fig.tight_layout()
        ax[0].set_title('Original Image')
        ax[0].imshow(img1)
        ax[1].set_title('Modified Image')
        ax[1].imshow(img2)
    
    def single_image_plots (self):
        pass
    
    def estimate_camera_intrinsic_paramters(self):
        
        self.read_images(img = self.image_1)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        
        corners_found, corners = cv.findChessboardCorners(image = gray,
                                                        patternSize=(9,6),
                                                        flags=None)

        if corners_found == True:
            self.world_3d_points.append(self.object_pts)
            corners2 = cv.cornerSubPix(image=gray,
                                        corners=corners,
                                        winSize=(11,11),
                                        zeroZone=(-1,-1), # A neglected zone. The -1,-1 is like None.
                                        criteria=criteria
                                        ) # more accurate corners (in subpixels)
            
            self.img_plane_2d_points.append(corners)
            cv.drawChessboardCorners(image=self.img,
                                    patternSize=(9,6),
                                    corners=corners2,
                                    patternWasFound=corners_found)
            
        image_size = gray.shape[::-1]
        # reprojection_error, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv.calibrateCamera(self.world_3d_points,
        #                                                                                                                     self.img_plane_2d_points,
        #                                                                                                                     image_size,
        #                                                                                                                     None,
        #                                                                                                                     None,
        #                                                                                                                     )
        # print(self.world_3d_points)
        # print(self.img_plane_2d_points)
    def calibration_objective(params, world_points, img_points):
        fx, fy, cx, cy = params
        intrinsic_matrix = np.array([[fx, 0, cx],
                                    [0, fy, cy],
                                    [0, 0, 1]])
        
        projected_points = np.dot(intrinsic_matrix, world_points.T).T

        error = np.linalg.norm(img_points - projected_points[:, :2], axis=1)

        return np.sum(error**2)
    
    def calculate_intrinsic_matrix_manually (self):
        '''
        The world_3d_points is a list instead of an array. All I do here is convert it to an array. The following method does the job.
        An alternative way is to simply extract the array from the list:
        
        self.world_3d_points[0]
        '''
        world_3d_pts = np.array(self.world_3d_points).reshape(-1,3)
        img_2d_pts = (np.array(self.img_plane_2d_points).reshape(-1,2))
        # Initial guess for intrinsic parameters
        initial_params = [1000, 1000, 1920 / 2, 1080 / 2]  # Adjust based on your image size

        # Perform optimization
        result = minimize(self.calibration_objective, initial_params, args=(world_3d_pts, img_2d_pts), method='leastsq')

        # Extract the optimized intrinsic parameters
        optimized_params = result.x
        optimized_fx, optimized_fy, optimized_cx, optimized_cy = optimized_params

        # Construct the final intrinsic matrix
        intrinsic_matrix_final = np.array([[optimized_fx, 0, optimized_cx],
                                        [0, optimized_fy, optimized_cy],
                                        [0, 0, 1]])

        print("Manually Calculated Intrinsic Matrix:")
        print(intrinsic_matrix_final)
        
        # world_3d_pts = np.array(self.world_3d_points).reshape(-1,3)
        # img_2d_pts = (np.array(self.img_plane_2d_points).reshape(-1,2))
        
        # min_len = min(len(world_3d_pts), len(img_2d_pts))
        # world_3d_pts = world_3d_pts[:min_len, :]
        # img_2d_pts = img_2d_pts[:min_len, :]
        
        # ones_col = np.ones((img_2d_pts.shape[0], 1))
        
        # img_2d_pts_aug = np.hstack((img_2d_pts, ones_col))
        
        
        # intrinsic_matrix, res, _,_ = np.linalg.lstsq(world_3d_pts,img_2d_pts_aug, rcond=None)
        # print(intrinsic_matrix)
        
        
if __name__ == '__main__':
    cam_calib = calibratePinholeCamera()
    cam_calib.estimate_camera_intrinsic_paramters()
    cam_calib.calculate_intrinsic_matrix_manually()