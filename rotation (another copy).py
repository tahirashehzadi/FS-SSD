import numpy as np
import cv2
import json
import os
from tqdm import tqdm

class RotationTransform:
    def __init__(self, height, width, angle):
        self.angle = angle
        self.height = height
        self.width = width
        self.center = (width / 2, height / 2)
        self.matrix = cv2.getRotationMatrix2D(self.center, angle, 1.0)

        # Calculate the new dimensions of the image
        abs_cos = abs(self.matrix[0, 0])
        abs_sin = abs(self.matrix[0, 1])
        bound_w = int(self.height * abs_sin + self.width * abs_cos)
        bound_h = int(self.height * abs_cos + self.width * abs_sin)

        # Adjust the rotation matrix to take into account translation
        self.matrix[0, 2] += bound_w / 2 - self.center[0]
        self.matrix[1, 2] += bound_h / 2 - self.center[1]

        self.new_size = (bound_w, bound_h)

    def apply_image(self, image):
        return cv2.warpAffine(image, self.matrix, self.new_size)

    def apply_coords(self, coords):
        # Add a column of ones to the coordinates matrix for the affine transformation
        ones = np.ones((coords.shape[0], 1))
        coords_ones = np.hstack([coords, ones])
        # Apply the rotation matrix
        rotated_coords = self.matrix.dot(coords_ones.T).T
        return rotated_coords[:, :2]

def transform_instance_annotations(anno, rotation_transform):
    if 'bbox' in anno:
        # Get the four corners of the bounding box
        x_min, y_min, width, height = anno['bbox']
        bbox = np.array([
            [x_min, y_min],  # top-left
            [x_min + width, y_min],  # top-right
            [x_min + width, y_min + height],  # bottom-right
            [x_min, y_min + height]  # bottom-left
        ])
        # Rotate the corners
        rotated_bbox = rotation_transform.apply_coords(bbox)
        # Calculate new bounding box coordinates
        x_min, y_min = rotated_bbox.min(axis=0)
        x_max, y_max = rotated_bbox.max(axis=0)
        anno['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
    
    if 'segmentation' in anno:
        new_segmentation = []
        for seg in anno['segmentation']:
            seg = np.array(seg).reshape(-1, 2)
            rotated_seg = rotation_transform.apply_coords(seg)
            new_segmentation.append(rotated_seg.flatten().tolist())
        anno['segmentation'] = new_segmentation

    return anno

def draw_annotations(image, annotations, rotation_transform):
    for anno in annotations:
        if 'bbox' in anno:
            bbox = anno['bbox']
            x_min, y_min, width, height = map(int, bbox)
            x_max = x_min + width
            y_max = y_min + height

            # Define the four corners of the bounding box
            corners = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
            
            # Apply rotation to bounding box corners
            rotated_corners = rotation_transform.apply_coords(corners)
            
            # Convert rotated corners to integer for drawing
            rotated_corners = np.round(rotated_corners).astype(int)
            
            # Draw the rotated bounding box
            for i in range(4):
                start_point = tuple(rotated_corners[i])
                end_point = tuple(rotated_corners[(i + 1) % 4])
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)

            # Draw the label (adjust position)
            label = anno.get('category_id', 'Unknown')
            label_text = anno.get('category_name', str(label))  # Use 'category_name' for textual labels
            
            # Find the position to place the label (e.g., near the top-left corner of the bounding box)
            text_position = (rotated_corners[0][0], rotated_corners[0][1] - 10)
            
            # Make sure the text position is within the image boundaries
            text_position = (max(text_position[0], 0), max(text_position[1], 0))
            
            # Draw the label text
            cv2.putText(image, label_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def process_rotated_bboxes(image_folder, annotations_file, output_annotation_file, output_annotated_image_folder, angle):
    # Load annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    rotated_annos = []

    # Create output directories if not exist
    if not os.path.exists(output_annotated_image_folder):
        os.makedirs(output_annotated_image_folder)

    # Iterate over each image in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Image file {image_file} not found or cannot be loaded.")
            continue
        
        # Initialize rotation transform
        h, w = image.shape[:2]
        rotation_transform = RotationTransform(h, w, angle)

        # Rotate the image
        rotated_image = rotation_transform.apply_image(image)

        # Apply rotation to annotations
        image_annotations = []
        for anno in data.get('annotations', []):
            if anno['image_id'] == data['images'][next(i for i, img in enumerate(data['images']) if img['file_name'] == image_file)]['id']:
                rotated_anno = transform_instance_annotations(anno, rotation_transform)
                image_annotations.append(rotated_anno)
                rotated_annos.append(rotated_anno)
        
        # Draw annotations on the rotated image
        annotated_image = draw_annotations(rotated_image.copy(), image_annotations, rotation_transform)
        
        # Save annotated image
        annotated_image_path = os.path.join(output_annotated_image_folder, f'annotated_{image_file}')
        cv2.imwrite(annotated_image_path, annotated_image)
    
    # Save updated annotations
    updated_data = {
        "images": data.get('images', []),
        "annotations": rotated_annos,
        "categories": data.get('categories', [])
    }
    
    with open(output_annotation_file, 'w') as f:
        json.dump(updated_data, f, indent=4)
    
    print(f"Updated annotations saved to {output_annotation_file}")
    print(f"Annotated images saved to {output_annotated_image_folder}")

def main():
    # Define paths
    image_folder = '/home/ifza/Documents/annotation/IMAGE'  # Folder with rotated images
    annotations_file = '/home/ifza/Documents/annotation/Annotation_P.json'  # Original annotations
    output_annotation_file = '/home/ifza/Documents/annotation/Rotated_BB_Annotations.json'  # Output annotations file
    output_annotated_image_folder = '/home/ifza/Documents/annotation/annotated_rotated'  # Folder to save annotated images
    angle = 15  # The angle used for rotation; should match the angle used in image rotation
    
    # Process rotated bounding boxes
    process_rotated_bboxes(image_folder, annotations_file, output_annotation_file, output_annotated_image_folder, angle)

if __name__ == "__main__":
    main()

