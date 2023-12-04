# Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    filtering_frame_arg = DeclareLaunchArgument(
        'filtering_frame', default_value='default_cam',
        description='The frame where the face pose filtering will take place')
    rgb_camera_arg = DeclareLaunchArgument(
        'rgb_camera', default_value='',
        description='The input camera namespace')
    rgb_camera_topic_arg = DeclareLaunchArgument(
        'rgb_camera_topic', default_value=[LaunchConfiguration('rgb_camera'), '/image_raw'],
        description='The input camera image topic')
    rgb_camera_info_arg = DeclareLaunchArgument(
        'rgb_camera_info', default_value=[LaunchConfiguration('rgb_camera'), '/camera_info'],
        description='The input camera info topic')

    face_detect_node = LifecycleNode(
        package='hri_face_detect', executable='face_detect', namespace='', name='hri_face_detect',
        parameters=[{'filtering_frame': LaunchConfiguration('filtering_frame')}],
        remappings=[
            ('image', LaunchConfiguration('rgb_camera_topic')),
            ('camera_info', LaunchConfiguration('rgb_camera_info'))])

    configure_event = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(face_detect_node),
        transition_id=Transition.TRANSITION_CONFIGURE))

    activate_event = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=face_detect_node, goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(face_detect_node),
            transition_id=Transition.TRANSITION_ACTIVATE))]))

    return LaunchDescription([
        filtering_frame_arg,
        rgb_camera_arg,
        rgb_camera_topic_arg,
        rgb_camera_info_arg,
        face_detect_node,
        configure_event,
        activate_event])
