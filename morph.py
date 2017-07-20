from face_landmark_detection import *

filename_jon = 'Faces\Jon.jpg';
filename_arya = 'Faces\Arya.jpg';

jon = cv2.imread(filename_jon);
arya = cv2.imread(filename_arya);

# Step 1: Get Landmarks
points_jon = get_facial_landmarks(filename_jon);
points_arya = get_facial_landmarks(filename_arya);

# Step 2: Delauney Triangulation
tri = Delaunay(points_jon);

jon_cut = jon.copy();
# Step 3: Warp Affine transform each triangle and paste it in the image
for tri_points_indices in tri.simplices.copy():
	# Image coordinates to be warped (Arya)
	orig_arya_pts = np.float32([points_arya[tri_points_indices,0], points_arya[tri_points_indices,1]]).transpose();
	to_be_warped_to = np.float32([points_jon[tri_points_indices,0], points_jon[tri_points_indices,1]]).transpose();

	# Compute the transformation matrix
	M = cv2.getAffineTransform(orig_arya_pts, to_be_warped_to);

	# Temporary transformed image (Arya)
	arya_assasin = cv2.warpAffine(arya, M, (arya.shape[1], arya.shape[0]));
	
	# Create Mask
	binary_mask = np.zeros((arya.shape[0], arya.shape[1]), dtype='int8');
	cv2.fillConvexPoly(binary_mask, np.array([points_jon[tri_points_indices,0], points_jon[tri_points_indices,1]], 'int32').transpose(), 1)

	# Cut the parts from jon picture
	jon_cut[binary_mask == 1] = arya_assasin[binary_mask == 1];

	# Keep seeing the updates
	cv2.imshow('lol', jon_cut);
	cv2.waitKey(10)


cv2.imshow('lol', jon_cut);
cv2.waitKey(1000);
cv2.imwrite('Results\Jon_Arya.jpg', jon_cut);
