#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode", # 是否开启无头模式，不显示仿真界面，服务器上训练时选择
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load") # 选择机械臂
parser.add_argument( # 若机械臂不是curobo自带的，需要给出外部资源路径
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument( # 是否只显示机械臂的碰撞球，这是机械臂在环境中的真实表示，避障时就是计算球与障碍物的距离
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument( # 是否使用MPC进行实时控制，False就是只进行一次点到点规划；True就是用MPC实时规划
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument( # 强制抓取时只能沿着z轴移动，且姿态不变，当RL求解出的动作太夸张时，可以启用这个约束
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose", # 修改和目标位姿的误差权重
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose", # 修改和当前位姿的误差权重
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)


args = parser.parse_args()

############################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920", # 仿真器分辨率
        "height": "1080",
    }
)
# Standard Library
from typing import Dict # 标准库类型提示

# Third Party
import carb # 打印日志或调整底层设置
import numpy as np
from helper import add_extensions, add_robot_to_scene # 作者自己写的库，添加插件，添加机械臂模型到场景
from omni.isaac.core import World # 创建仿真物理世界
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction # 打包控制指令发送给机器人

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType # 碰撞检测类型
from curobo.geom.types import WorldConfig # 环境障碍物配置
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose # 机械臂位姿类
from curobo.types.robot import JointState # 机械臂关节状态类
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger # 日志打印
from curobo.util.usd_helper import UsdHelper # 读取usd场景文件
from curobo.util_file import ( # 文件路径工具
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path, # 机器人配置文件路径
    get_world_configs_path, # 环境障碍物配置文件路径
    join_path,
    load_yaml, # 读取yaml文件并返回字典
)
from curobo.wrap.reacher.motion_gen import ( # MPC求解器类
    MotionGen,
    MotionGenConfig, # MPC求解器配置参数
    MotionGenPlanConfig,
    PoseCostMetric, # 代价函数
)

############################################################


########### OV #################;;;;;


def main():
    # create a curobo motion gen instance:
    num_targets = 0 # 记录已经成功到达目标的次数
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0) # 创建仿真物理世界（定义单位是米）
    stage = my_world.stage # stage管理场景中的所有物体

    xform = stage.DefinePrim("/World", "Xform") # 定义场景的根节点
    stage.SetDefaultPrim(xform) # 将world设为默认根节点
    stage.DefinePrim("/curobo", "Xform") # 定义curobo根节点
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid( # 定义目标方块，VisualCuboid表示只有视觉效果没有碰撞属性，机械臂可以穿过它
        "/World/target", # 定义在场景图中的路径
        position=np.array([0.5, 0, 0.5]), # 初始位置
        orientation=np.array([0, 1, 0, 0]), # 初始姿态，四元数表示
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn") # 设置日志打印等级为warn
    past_pose = None # 记录上一个时间步目标位置
    n_obstacle_cuboids = 30 # 预留方块障碍物的存储空间
    n_obstacle_mesh = 100 #预留网格的存储空间

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType() # 定义tensor的设备类型，这里是GPU
    # 在curobo中加载机械臂配置文件
    robot_cfg_path = get_robot_configs_path() # 机器人配置文件.yml路径
    if args.external_robot_configs_path is not None: # 检测是否使用了外部机械臂配置文件
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"] # 机械臂关节名称列表
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"] # 机械臂默认收回位置

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world) # 将机械臂模型添加到场景中

    articulation_controller = None

    # 在curobo中加载环境障碍物配置文件
    world_cfg_table = WorldConfig.from_dict( # 加载环境障碍物文件
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02 # 调整障碍物高度
    world_cfg1 = WorldConfig.from_dict( # 加载mesh障碍物
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh" # 修改mesh名称，避免和cuboid重复
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # 配置优化算法参数
    trajopt_dt = None # reactive=False时，优化步长trajopt_dt可以变化（optimize_dt就是规定优化步长能不能变化的布尔值），求解优化轨迹时使用MPPI方法
    optimize_dt = True
    trajopt_tsteps = 32 # 预测步长，和优化步长一起组成了一条完整轨迹
    trim_steps = None
    max_attempts = 4 # 寻找轨迹最多尝试4次
    interpolation_dt = 0.05 # 插值步长，真正控制步长
    enable_finetune_trajopt = True # 精修轨迹，使其更平滑
    if args.reactive: # reactive=True时，用MPC方法，优化步长固定且等于插值步长，每一步优化也使用MPPI方法
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None] # 只利用第一步控制指令
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False
    motion_gen_config = MotionGenConfig.load_from_robot_config( # 将上面的参数赋给MPC求解器
        robot_cfg, # 机械臂动力学
        world_cfg, # 环境障碍物
        tensor_args, # 设置GPU
        collision_checker_type=CollisionCheckerType.MESH, # 采用mesh碰撞检测类型
        num_trajopt_seeds=12, # MPPI采样的种子数
        num_graph_seeds=12, # 图搜索采样的种子数（越大优化的越准但算的越慢）
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt, # 离散化时间步长
        trajopt_tsteps=trajopt_tsteps, # 预测时域
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config) # 创建求解器实例
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False) # 预热求解器，目的是将cuda内核编译、显存分配、缓存和一些内部结构初始化好，避免第一次规划时卡顿。
        #warmup_js_trajopt是预热关节空间的变量，若为False就只预热笛卡尔空间的规划（机械臂末端位姿）

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode) # 根据headless_mode参数加载插件

    plan_config = MotionGenPlanConfig( # 实际运行时的求解器参数
        enable_graph=False, # 禁用图搜索
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0, # 时间膨胀因子，本来规划t1秒到达A点，实际中只需t1/time_dilation_factor秒到达就行了，只适用于离线规划场景，不适用于MPC和RL
    )

    usd_help.load_stage(my_world.stage) # 将isaac中的stage传给curobo
    usd_help.add_world_to_stage(world_cfg, base_frame="/World") # 将curobo中的障碍物添加到isaac的stage中

    cmd_plan = None # 记录MPC优化出的轨迹
    cmd_idx = 0 # 记录当前执行到cmd_plan的第几个控制指令
    my_world.scene.add_default_ground_plane() # 给物理世界添加地面
    i = 0
    spheres = None
    past_cmd = None # 记录上一个时间步的控制指令
    target_orientation = None # 记录目标姿态
    past_orientation = None # 记录上一时刻的姿态
    pose_metric = None # 存储代价函数
    while simulation_app.is_running(): # 判断仿真器是否关闭
        my_world.step(render=True) # 物理世界前进一步，并同步显示在仿真界面上
        if not my_world.is_playing(): # 判断仿真器的play按钮是否按下
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index # 获取当前时间步索引
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller() # 获取控制器句柄
        # if step_index < 10:
        if step_index < 2:
            my_world.reset()
            robot._articulation_view.initialize() # 初始化关节视图
            idx_list = [robot.get_dof_index(x) for x in j_names] # 机械臂关节索引列表
            robot.set_joint_positions(default_config, idx_list) # 将机械臂移动到默认收回位置

            robot._articulation_view.set_max_efforts( # 设置关节电机最大力矩
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0: # 更新MPC求解器中的障碍物信息
            print("Updating world, reading w.r.t.", robot_prim_path) # 打印机器人路径
            obstacles = usd_help.get_obstacles_from_stage( # 获取障碍物信息，排除机器人本体、目标和地面
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(len(obstacles.objects)) # 打印障碍物数量

            motion_gen.update_world(obstacles) # 把环境障碍物信息同步给MPC求解器
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose() # 获取目标的位置和姿态

        if past_pose is None: # 当前和上一步目标位姿初始化
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state() # 从仿真器中获取机械臂当前关节状态
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        if np.any(np.isnan(sim_js.positions)): # NAN检查，防止数值爆炸
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions), # 转成GPU tensor
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0, # 加速度的导数
            joint_names=sim_js_names,
        )

        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None: # 延迟补偿，直接使用下一时刻应该到达的位置来计算下一步动作
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names) # 对齐关节顺序

        if args.visualize_spheres and step_index % 2 == 0: # 可视化机械臂碰撞球
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:

                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))

        robot_static = False
        if (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive:
            robot_static = True
        if ( # 判断是否需要重置目标重新规划
            (
                np.linalg.norm(cube_position - target_pose) > 1e-3
                or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
            )
            and np.linalg.norm(past_pose - cube_position) == 0.0
            and np.linalg.norm(past_orientation - cube_orientation) == 0.0
            and robot_static
        ):
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation

            # compute curobo solution:
            ik_goal = Pose( # 目标转换成tensor形式
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric # 加载代价函数配置
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config) # 一次MPC优化，输出为关节状态
            # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

            succ = result.success.item()  # ik_result.success.item()
            if num_targets == 1:
                if args.constrain_grasp_approach:
                    pose_metric = PoseCostMetric.create_grasp_approach_metric()
                if args.reach_partial_pose is not None:
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            if succ:
                num_targets += 1
                cmd_plan = result.get_interpolated_plan() # 获取规划出的关节状态
                cmd_plan = motion_gen.get_full_js(cmd_plan) # 转换成完整的关节状态（包含固定和被动关节）
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx] # 获取当前MPC优化的第cmd_idx个控制指令
            past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction( # 把GPU控制输入数据传回CPU
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action) # 将控制指令发送给电机
            cmd_idx += 1
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


if __name__ == "__main__":
    main()
