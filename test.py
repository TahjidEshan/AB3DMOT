import cv2, sys
import numpy as np
from PIL import Image
from kitti_utils import read_label, compute_box_3d, draw_projected_box3d, Calibration, Object3d, load_velo_scan, compute_orientation_3d
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

if __name__ == "__main__":
    print("Starting")
    image_path =  '/home/eshan/Pictures/000009.jpeg'
    img = cv2.imread(image_path)
    file_path = '/home/eshan/Pictures/0001.txt'
    filecontent = np.array([f.split() for f in open(file_path, 'r')])
    alllabels = np.unique(filecontent[:,2])
    labels = ['Car', 'Pedestrian', 'Cyclist']
    finalset = [x for x in alllabels if x not in labels]
    for val in finalset:
        filecontent = filecontent[filecontent[:,2]!=val,:]
    data = (Object3d(getstringfromarray(line[2:])) for line in filecontent[filecontent[:,0] == '0',:])
    calib = Calibration('/home/eshan/Pictures/calib.txt')	
    velo = load_velo_scan('/home/eshan/Pictures/000009.bin', np.float32, n_vec=4)[:,0:4]
    img2 = np.copy(img)
    show_lidar_with_boxes(velo, data, calib)
    for obj in data:
        print("here")
        box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
        img2 = draw_projected_box3d(img2, box3d_pts_2d, color=(0,0,0))
        text = 'gt ID: %d, Type: %s' % (0, obj.type)
        if box3d_pts_2d is not None:
            print("also")
            img2 = cv2.putText(img2, text, (int(box3d_pts_2d[4, 0]), int(
                box3d_pts_2d[4, 1]) - 8), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color=(0,0,0))
    # img = Image.fromarray(img2)
    # img = img.resize((width, height))
    cv2.imshow("image", img2)
    cv2.waitKey()