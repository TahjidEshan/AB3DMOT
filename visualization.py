import os
import numpy as np
import sys
import cv2
from PIL import Image
from utils import is_path_exists, mkdir_if_missing, load_list_from_folder, fileparts, random_colors
from kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration, Object3d, load_velo_scan, compute_orientation_3d

max_color = 30
colors = random_colors(max_color)       # Generate random colors
type_whitelist = ['Car', 'Pedestrian', 'Cyclist']
score_threshold = -10000
width = 1242
height = 374
# seq_list = ['0000', '0001','0002', '0003','0004', '0005','0006', '0007','0008', '0009','0010', '0011','0012', '0013','0014', '0015','0016', '0017','0018', '0019','0020']
seq_list =  ['0000']
def getstringfromarray(line):
    l = ""
    for i in line:
        l = l+i+" "
    return l.strip()
def show_lidar_with_boxes(
    pc_velo,
    objects,
    calib,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color, label=obj.type)

        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))
        #

        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)
def vis(result_sha, data_root, result_root):
    def show_image_with_boxes(img,velo, objects_res, objects_res_det,objects_res_raw, labeldata,  object_gt, calib, save_path, height_threshold=0):
        img2 = np.copy(img)

        for obj in objects_res:
            box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
            color_tmp = tuple([int(tmp * 255)
                               for tmp in colors[obj.id % max_color]])
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0,0,255))
            text = 'Tracked ID: %d, Type: %s' % (obj.id, obj.type)
            if box3d_pts_2d is not None:
                img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(
                    box3d_pts_2d[4, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(0,0,255))
        for obj in objects_res_det:
            box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
            color_tmp = tuple([int(tmp * 255)
                               for tmp in colors[obj.id % max_color]])
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0,255,0))
            text = 'Detection ID: %d, Type: %s' % (obj.id, obj.type)
            if box3d_pts_2d is not None:
                img2 = cv2.putText(img2, text, (int(box3d_pts_2d[3, 0]), int(
                    box3d_pts_2d[3, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(0,255,0))
        for obj in labeldata:
            # print("here")
            box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
            img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(255,0,0))
            text = 'GT, Type: %s' % (obj.type)
            if box3d_pts_2d is not None:
                # print("also")
                print(text)
                img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(
                    box3d_pts_2d[4, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(255,0,0))
        # for obj in objects_res_raw:
        #     box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
        #     color_tmp = tuple([int(tmp * 255)
        #                        for tmp in colors[obj.id % max_color]])
        #     img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(255,0,0))
        #     text = 'Estimate ID: %d' % obj.id
        #     if box3d_pts_2d is not None:
        #         img2 = cv2.putText(img2, text, (int(box3d_pts_2d[2, 0]), int(
        #             box3d_pts_2d[2, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(255,0,0))
        show_lidar_with_boxes(velo, labeldata, calib, objects_pred=objects_res)
        img = Image.fromarray(img2)
        img = img.resize((width, height))
        #cv2.imshow("Image", img)
        print("Saving Image at", save_path)
        # img.save(save_path)

    for seq in seq_list:
        image_dir = os.path.join(data_root, 'image_02/%s' % seq)
        calib_file = os.path.join(data_root, 'calib/%s.txt' % seq)
        label_file = os.path.join(data_root, 'label_02/%s.txt' % seq)
        velo_dir = os.path.join(data_root, 'velodyne/%s' % seq)
        result_dir = [os.path.join(
            result_root, '%s/trk_withid/%s' % (result_sha[0], seq)),os.path.join(
            result_root, '%s/trk_withid/%s' % (result_sha[1], seq)),os.path.join(
            result_root, '%s/trk_withid/%s' % (result_sha[2], seq))]
        save_3d_bbox_dir = os.path.join(
            result_root, '%s/trk_image_vis/%s' % ("Combined_Final_WithLabel", seq))
        mkdir_if_missing(save_3d_bbox_dir)

        # load the list
        images_list, num_images = load_list_from_folder(image_dir)
        velo_list, num_velo=load_list_from_folder(velo_dir)
        print('number of images to visualize is %d' % num_images)
        start_count = 0
        filecontent = np.array([f.split() for f in open(label_file, 'r')])
        # alllabels = np.unique(filecontent[:,2])
        # labels = ['Car', 'Pedestrian', 'Cyclist']
        # finallabelset = [x for x in alllabels if x not in labels]
        # print(alllabels)
        # print(finallabelset)
        # for val in finallabelset:
        #     filecontent = filecontent[filecontent[:,2]!=val,:]
        # print(np.unique(filecontent[:,2]))
        for count in range(start_count, num_images):
            image_tmp = images_list[count]
            velo_tmp = velo_list[count]
            if not is_path_exists(image_tmp):
                count += 1
                continue
            image_index = int(fileparts(image_tmp)[1])
            image_tmp = np.array(Image.open(image_tmp))
            img_height, img_width, img_channel = image_tmp.shape
            filecontentframe =  filecontent[filecontent[:,0] == str(image_index),:]
            print(f"Labels for frame {image_index}",np.unique(filecontentframe[:,2]))
            labeldata = (Object3d(getstringfromarray(line[2:])) for line in filecontentframe)
            # result_tmp = os.path.join(
            #     result_dir, '%06d.txt' % image_index)		# load the result
            # if not is_path_exists(result_tmp):
            #     object_res = []
            # else:
            #     object_res = read_label(result_tmp)
            # result_tmp_det = os.path.join(
            #     result_dir, 'det%06d.txt' % image_index)		# load the result
            # if not is_path_exists(result_tmp_det):
            #     object_res_det = []
            # else:
            #     object_res_det = read_label(result_tmp_det)      
            # result_tmp_raw = os.path.join(
            #     result_dir, 'raw%06d.txt' % image_index)		# load the result
            # if not is_path_exists(result_tmp_raw):
            #     object_res_raw = []
            # else:
            #     object_res_raw = read_label(result_tmp_raw)          
            object_res = []
            object_res_det = []
            object_res_raw = []
            for dirt in result_dir:
                result_tmp = os.path.join(
                    dirt, '%06d.txt' % image_index)		# load the result
                if is_path_exists(result_tmp):
                    object_res = object_res + read_label(result_tmp)
                result_tmp_det = os.path.join(
                    dirt, 'det%06d.txt' % image_index)		# load the result
                if is_path_exists(result_tmp_det):
                    object_res_det = object_res_det + read_label(result_tmp_det)      
                result_tmp_raw = os.path.join(
                    dirt, 'raw%06d.txt' % image_index)		# load the result
                if is_path_exists(result_tmp_raw):
                    object_res_raw = object_res_raw + read_label(result_tmp_raw)          
            print('processing index: %d, %d/%d, results from %s' %
                  (image_index, count+1, num_images, result_tmp))
            calib_tmp = Calibration(calib_file)			# load the calibration

            object_res_filtered = []
            for object_tmp in object_res:
                if object_tmp.type not in type_whitelist:
                    continue
                if hasattr(object_tmp, 'score'):
                    if object_tmp.score < score_threshold:
                        continue
                center = object_tmp.t
                object_res_filtered.append(object_tmp)
            object_res_filtered_det = []
            for object_tmp in object_res_det:
                if object_tmp.type not in type_whitelist:
                    continue
                if hasattr(object_tmp, 'score'):
                    if object_tmp.score < score_threshold:
                        continue
                center = object_tmp.t
                object_res_filtered_det.append(object_tmp)
            object_res_filtered_raw = []
            for object_tmp in object_res_raw:
                if object_tmp.type not in type_whitelist:
                    continue
                # if hasattr(object_tmp, 'score'):
                #     if object_tmp.score < score_threshold:
                #         continue
                center = object_tmp.t
                object_res_filtered_raw.append(object_tmp)
            num_instances = len(object_res_filtered)
            save_image_with_3dbbox_gt_path = os.path.join(
                save_3d_bbox_dir, '%06d.jpg' % (image_index))
            velodyne_scan = load_velo_scan(velo_tmp, np.float32, n_vec=4)[:,0:4]
            show_image_with_boxes(image_tmp, velodyne_scan, object_res_filtered, object_res_filtered_det,object_res_filtered_raw,labeldata, [
            ], calib_tmp, save_path=save_image_with_3dbbox_gt_path)
            print('number of objects to plot is %d, %d, %d' % (num_instances, len(object_res_filtered_det), len(object_res_filtered_raw)))
            count += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualization.py result_sha(e.g., car_3d_det_test)")
        sys.exit(1)

    result_root = '/content/gdrive/My Drive/Colab Notebooks/AB3DMOT/results'
    result_sha = sys.argv[1]
    results = ["car_3d_det_val", "ped_3d_det_val", "cyc_3d_det_val"]
    if 'val' in result_sha:
        data_root = '/content/gdrive/My Drive/Dataset/KITTI/kitti_tracking/training'
    elif 'test' in result_sha:
        data_root = '/content/gdrive/My Drive/Dataset/KITTI/kitti_tracking/testing'
    else:
        print("wrong split!")
        sys.exit(1)

    vis(results, data_root, result_root)
