# -- coding: utf-8 --
# @Time : 2022/11/7
# @Author : ykk648
# @Project : https://github.com/ykk648/apstone

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from cv2box import CVImage

model_id = 'damo/cv_hrnetv2w32_body-2d-keypoints_image'
body_2d_keypoints = pipeline(Tasks.body_2d_keypoints, model=model_id)
output = body_2d_keypoints('resource/for_pose/t_pose_1080p.jpeg')

# the output contains poses, scores and boxes
print(output)
CVImage(CVImage('resource/for_pose/t_pose_1080p.jpeg').draw_landmarks(output['keypoints'][0])).show()
