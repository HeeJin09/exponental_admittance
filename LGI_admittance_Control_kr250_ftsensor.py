from re import L
import pybullet as p
import lie_robotics as lr
import numpy as np
# np.set_printoptions(precision=3, suppress=True)
import scipy.io
import time
import json
from numpy.linalg import inv, pinv, LinAlgError
import matplotlib.pyplot as plt
import modern_robotics as mr

def getJointStates(robotId, joint_list):
    jointState = p.getJointStates(robotId, joint_list)
    q = []
    qdot = []
    for i in range(0, len(joint_list)):
        js = jointState[i]
        q.append(js[0])
        qdot.append(js[1])
    q = np.array(q)
    qdot = np.array(qdot)
    return q, qdot

def LieScrewTrajectory(X0, XT, V0, VT, dV0, dVT, Tf, N):
    Xd_list = []
    Vd_list = []
    dVd_list = []
    
    lambda_0 = np.zeros(6)
    lambda_T = lr.se3ToVec(lr.MatrixLog6(lr.TransInv(X0) @ XT))
    dlambda_0 = V0
    dlambda_T = lr.dlog6(-lambda_T) @ VT
    ddlambda_0 = dV0
    ddlambda_T = lr.dlog6(-lambda_T) @ dVT + lr.ddlog6(-lambda_T, -dlambda_T) @  VT

    timegap = Tf / (N - 1.0)
    Xd_list, Vd_list, dVd_list = [], [], []

    for i in range(N):
        lambda_t = np.zeros(6)
        dlambda_t = np.zeros(6)
        ddlambda_t = np.zeros(6)
        t = timegap * (i - 1)
        for j in range(6):
            ret = lr.QuinticTimeScalingKinematics(lambda_0[j], lambda_T[j], dlambda_0[j], dlambda_T[j],
                                                ddlambda_0[j], ddlambda_T[j], Tf, t)
            lambda_t[j] = ret[0]
            dlambda_t[j] = ret[1]
            ddlambda_t[j] = ret[2]

        V = np.dot(lr.dexp6(-lambda_t), dlambda_t)
        dV = np.dot(lr.dexp6(-lambda_t), ddlambda_t) + np.dot(lr.ddexp6(-lambda_t, -dlambda_t), dlambda_t)
        T = X0 @ lr.MatrixExp6(lr.VecTose3(lambda_t))
        Xd_list.append(T)
        Vd_list.append(V)
        dVd_list.append(dV)

    return Xd_list, Vd_list, dVd_list
def homogeneous_Transform(boxId, link_index):
	link_state = p.getLinkState(boxId, linkIndex=link_index)
	link_position = link_state[0]
	link_orientation = link_state[1]
	X_des_= np.eye(4)
	rotation_matrix = p.getMatrixFromQuaternion(link_orientation)
	rotation_matrix = np.array(rotation_matrix).reshape((3, 3))
	X_des_[:3, 3] = link_position
	X_des_[:3, :3] = rotation_matrix
	return X_des_

def lineDebug_function(X, Teef_tool, x_lineId, y_lineId, z_lineId):
	
	p0= X@Teef_tool@np.array([0,0,0,1])
	p0x= X@Teef_tool@np.array([0.3,0,0,1])
	p0y= X@Teef_tool@np.array([0,0.3,0,1])
	p0z= X@Teef_tool@np.array([0,0,0.3,1])	
	p.addUserDebugLine(p0[0:3],p0x[0:3],[1,0,0],replaceItemUniqueId=x_lineId)
	p.addUserDebugLine(p0[0:3],p0y[0:3],[0,1,0],replaceItemUniqueId=y_lineId)
	p.addUserDebugLine(p0[0:3],p0z[0:3],[0,0,1],replaceItemUniqueId=z_lineId)
 
      
def Force_Torque_Sensor_value(robotId, index, Tsensor_eef, i, prev_sensor_Wrench) :
	sensor_Wrench = np.array(p.getJointState(robotId, index)[2])

	if i == 0:
		Wrench_init = np.array([0,0,0,0,0,0])
	else:
		Wrench_init = [init_force[0], init_force[1], init_force[2], init_torque[0], init_torque[1], init_torque[2]]
	AdTsensor_eef =  lr.Adjoint(Tsensor_eef)
	F_eef_mass = np.transpose(AdTsensor_eef) @  sensor_Wrench
	
	Wrench =  (F_eef_mass - Wrench_init)
    
	fileter_sensor_Wrench = alpha*(Wrench)+(1-alpha)*prev_sensor_Wrench
	prev_sensor_Wrench = fileter_sensor_Wrench
	return fileter_sensor_Wrench
############# Load URDF and setting ############
p.connect(p.GUI)
p.setGravity(0,0,0)
p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=0.0, cameraPitch=-0, cameraTargetPosition=[1.5, 0, 1])
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  

endtime = 4
CONTROL_FREQ = 1000.0
dt = 1 / CONTROL_FREQ
N = int(endtime/dt)

####################load URDF #####################
robotId  =p.loadURDF("urdf/kuka_kr250_r2700.urdf", )
boxId  =p.loadURDF("urdf/cylinder.urdf", [6+ 1.30018632, 0,    2.12461829],[0,0,0,1])
boxId2  =p.loadURDF("urdf/cylinder.urdf", [7+ 1.30018632, 0,    2.12461829], [0,0,0,1])


file_path = "kuka_kr250_2700.json"

with open(file_path, 'r') as file:
    data = json.load(file)

M = np.array(data['M'])
Slist =  np.array(data['Slist'])
Mlist =  np.array(data['Mlist'])
Glist =  np.array(data['Glist'])
Blist =  np.array(data['Blist'])
print("M", M)
print("Mlist", Mlist)
print("Slist . : ", Slist)
print("Blist . : ", Blist)
############### robot initial setting ##############
joint_list = [1, 2, 3, 4, 5, 6]
joint_State = [0, 0, 0, 0, 0, 0]
for i in joint_list:
    p.setJointMotorControl2(robotId, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)              

############ Sensor ON #######################
# p.enableJointForceTorqueSensor(robotId, 8, 1)
init_sensor_Wrench= np.array([0,0,0,0,0,0])
frq_cutoff = 30
alpha = (frq_cutoff*(1/CONTROL_FREQ))/(1+frq_cutoff*(1/CONTROL_FREQ))
prev_sensor_Wrench = np.array([0,0,0,0,0,0])
############# parameter initial setting ###############
Ftip = np.zeros(6)
F_des = np.transpose([0,0,0,0,0,0])
F_ext = [0,0,0]
T_ext = [0,0,0] 

eef_mass = 0.01; #kg
qd=np.array([0,0, 0, 0, 0,0])
init_force = (np.array([0.0,0.0, eef_mass* (0)])) 
init_torque = np.array([0.0, 0.0, 0.0])
Wrench_init = [init_force[0], init_force[1], init_force[2], init_torque[0], init_torque[1], init_torque[2]]

Tsensor_eef = np.array([[1 ,0 ,0,0.05],[0, 1 ,0,0],[0 ,0 ,1,0],[0 ,0 ,0,1]])
thetalist, dthetalist = getJointStates(robotId, joint_list)
inti_Trobot_end = lr.FKinSpace(M,Slist,thetalist)
inti_Trobot_rotation = inti_Trobot_end[0:3,0:3]

t = 0
################# Admittance control parameter #######################
# Gain
A = 1*np.diag([1,1,1,1,1,1])
D = 22*np.diag([1,1,1,1,1,1])
K = 100*np.diag([1,1,1,1,1,1])

Kp = np.diag([0,0,0,0,0,0])
Kv = np.diag([0,0,0,0,0,0])
Kr = np.diag([0,0,0,0,0,0])

############## Trajectory
X_start = lr.FKinBody(M, Blist, thetalist)
X_q = np.array(X_start)
# X_q[0,3] = X_start[0,3] + 0.3
# X_q[1,3] = X_start[1,3] + 0.1
print(X_start)
X_end = X_q 
R_start = lr.TransToR(X_start)
P_start = lr.TransToP(X_start)

V_start = np.transpose([0,0,0,0,0,0])
V_end = np.transpose([0,0,0,0,0,0])
dV_start = np.transpose([0,0,0,0,0,0])
dV_end = np.transpose([0,0,0,0,0,0])

Xd_list=[]
Vd_list=[]
dVd_list=[]
t_list = []
lambda_des_list = []
lambda_act_list = []

integral_lambda = np.transpose([0,0,0,0,0,0])
integral_F_err = np.transpose([0,0,0,0,0,0])
pre_lambda = np.transpose([0,0,0,0,0,0])
pre_F_err = np.transpose([0,0,0,0,0,0])
pre_q_com = np.array([0,0,0,0,0,0])
dq_ref = np.array([0,0,0,0,0,0])

for _ in range(N + 1):
    Xd_list.append(np.zeros((4, 4)))
    Vd_list.append(np.zeros(6))
    dVd_list.append(np.zeros(6))

Xd_list,Vd_list,dVd_list= LieScrewTrajectory(X_start,X_end,V_start,V_end,dV_start,dV_end,endtime,int(endtime/dt))




############# plot parameter ############
x_lineId = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                            lineToXYZ=[0, 0, 0],
                            lineColorRGB=[1,0,0],
                            lineWidth=1,
                            lifeTime=dt*200)
y_lineId = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                            lineToXYZ=[0, 0, 0],
                            lineColorRGB=[0,1,0],
                            lineWidth=1,
                            lifeTime=dt*200)
z_lineId = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                            lineToXYZ=[0, 0, 0],
                            lineColorRGB=[0,1,0],
                            lineWidth=1,
                            lifeTime=dt*200) 

Teef_tool = np.eye(4)
# Teef_tool[0,3]=0.155
Teef_tool[0,3]=0.155
Teef_tool[1,3]=0.0
Teef_tool[2,3]=0.0
d = 0.05
pos = [0,0,0]

Tsb = np.eye(4)
# Teef_tool[0,3]=0.155
Tsb[0,3]=0.15
Tsb[1,3]=0.0
Tsb[2,3]=0.0
############## Free body inertia ###########
A_inertia = np.eye(6)
Inertia_Matrix = np.eye(3)
Inertia_Matrix[0,0]= 12.3333333333
Inertia_Matrix[1,1]= 12.33333333
Inertia_Matrix[2,2]= 24.5

A_inertia[3:6, 3:6] = Inertia_Matrix 

A_inertia[:3, :3] = eef_mass* np.eye(3)
V_ref = np.array([0,0,0,0,0,0])
V_ref_ = np.array([0,0,0,0,0,0]) 

test = np.array([0,0,0,0,0,0])
link_index = 0
# X_des_ = np.array(homogeneous_Transform(boxId, link_index))

for i in range(N):
	thetalist, dthetalist = getJointStates(robotId, joint_list)

	################# Apply external force/torque ###################
	### External body wrench ####
	if i == 500:
		pos = [0.15,0.4, 0.4]
		F_ext = [-100,0, 10]
		# F_ext = [0,0,0]
		T_ext = np.cross(pos, F_ext)
	else:
		pos = [0,0,0]
		F_ext = [0,0,0]
		T_ext = np.cross(pos, F_ext)
		
	# p.applyExternalForce(robotId, 8, F_ext, pos, p.LINK_FRAME)
	# p.applyExternalTorque(robotId, 8, T_ext, p.LINK_FRAME)
	Wrench_ext = np.array([-F_ext[0], -F_ext[1], -F_ext[2], -T_ext[0], -T_ext[1], -T_ext[2]])

	############ Calculate sensor and actual wrench  #################
	# fileter_sensor_Wrench = Force_Torque_Sensor_value(robotId, 7, Tsensor_eef, i, prev_sensor_Wrench)
	# prev_sensor_Wrench = fileter_sensor_Wrench
	# fileter_sensor_Wrench = np.round(fileter_sensor_Wrench, 1) 
	# sensor_Wrench = np.array(p.getJointState(robotId, 8)[2])
	##############Contact Poitn ##################
	f = np.array([Wrench_ext[0],Wrench_ext[1],Wrench_ext[2]])
	norm_f= np.linalg.norm(f) + 0.0000000000000001

	u = f/norm_f
	n = np.array([1,0,0]).T


	U = np.array([[0 ,-u[2] ,u[1]],[u[2] ,0 ,-u[0]],[-u[1], u[0], 0],[n[0], n[1], n[2]]])
	b =	np.transpose([-Wrench_ext[3]/norm_f,-Wrench_ext[4]/norm_f,-Wrench_ext[5]/norm_f,d ])
	try:
		r = np.linalg.inv(U.T @ U)@U.T @ b
	except:
		r = [0,0,0]

	####################### Trajectory ##########################
	X_des = Xd_list[i]
	V_des = (Vd_list[i])
	dV_des = (dVd_list[i])
	
	X_test = X_des @ np.transpose(Tsb)
	lambda_des = lr.se3ToVec(lr.MatrixLog6(X_des))
	R_des = lr.TransToR(X_des)
	P_des = lr.TransToP(X_des)

	################ FK&IK ###############################
	X = lr.FKinBody(M,Blist,thetalist)	
	invX = lr.TransInv(X)
	Js = lr.JacobianSpace(Slist, thetalist)
	Jb = lr.JacobianBody(Blist, thetalist)

	V = Jb @ dthetalist
	
	R_act = lr.TransToR(X)
	P_act = lr.TransToP(X)
	p.applyExternalForce(boxId, -1, F_ext, np.array([pos[0]+0.15, pos[1], pos[2]]), p.LINK_FRAME)
	p.applyExternalForce(boxId2, -1, F_ext, pos, p.LINK_FRAME)
        
	homogeneous_transform = np.array(homogeneous_Transform(boxId, 0))
	homogeneous_transform_ = homogeneous_transform
	homogeneous_transform_[0,3] = homogeneous_transform[0,3]- 6

	homogeneous_transform1 = np.array(homogeneous_Transform(boxId2, 0))
	homogeneous_transform_1 = homogeneous_transform1
	homogeneous_transform_1[0,3] = homogeneous_transform_1[0,3]- 7
	##################### Err ###########################

	X_err = invX @ homogeneous_transform_
	invX_err = lr.TransInv(X_err)
	V_err = V_des - (lr.Adjoint(invX_err) @ V)
	F_err = 0.5*(F_des - lr.Adjoint(X_err) @ Wrench_ext)
	lambda_ = lr.se3ToVec(lr.MatrixLog6(X_err))
	dlambda = lr.dlog6(-lambda_) @ V_err
	gamma = np.transpose(lr.dexp6(-lambda_)) @ F_err

	################ Gain test ######################### 
	KV = lr.dlog6(-lambda_) @ inv(A) @ D @ lr.dexp6(-lambda_)
	KP = lr.dlog6(-lambda_) @ inv(A) @ K 
	KG = lr.dlog6(-lambda_)
	
	integral_lambda = integral_lambda + (lambda_ *dt)
	integral_F_err = integral_F_err + (F_err*dt)
	# dlambda_ref = -dlambda - KV @ lambda_ - KP @ (integral_lambda) + KG @ integral_F_err
	dlambda_ref = -dlambda - KV @ lambda_ - KP @ (integral_lambda) 

	lambda_act = lr.se3ToVec(lr.MatrixLog6( invX @ homogeneous_transform1))
	#################################################
	# dV_ref = inv(A_inertia) @ (((-1*(A_inertia @ lr.ad(V)) + (np.transpose(lr.ad(V)) @ A_inertia)) @ V + Wrench_ext ))
	# V_ref_ = V_ref_ + dV_ref * dt

	#################################################
	V_ref = lr.Adjoint(X_err) @ (V_des - lr.dexp6(-lambda_) @ dlambda_ref)
	
	test = lambda_
	############damped jacobian method #############
	# dq_ref = inv(Jb + (np.identity(6)*0.0001)) @ V_ref
	
	##########remove singular pose method
	dq_ref = pinv(Jb) @  (V_ref)	
	q_com = pre_q_com + (dq_ref * dt)
	pre_q_com = q_com 
	############# Position Control ###################
	p.setJointMotorControlArray(robotId, joint_list, p.POSITION_CONTROL ,targetPositions = q_com)  
	#################################################
	
	r_test = np.eye(4)
	r_test[0,3]=pos[0] + 0.005
	r_test[1,3]=pos[1]
	r_test[2,3]=pos[2]
	# lineDebug_function(X, r_test, x_lineId, y_lineId, z_lineId, dt)

	################ Debug line ######################
	lineDebug_function(X, Teef_tool, x_lineId, y_lineId, z_lineId)

	
	########### plot parameter update and save ##########
	#### Save data ####
	t = t + dt
	t_list.append(t)
	lambda_des_list.append(lambda_)	
	lambda_act_list.append(lambda_act)	
	p.stepSimulation()
	time.sleep(dt)


ax = plt.subplot(111)
plt.plot(t_list, np.array([elem[0] for elem in lambda_act_list]), label='Eta_x')
plt.plot(t_list, np.array([elem[1] for elem in lambda_act_list]), label='Eta_y')
plt.plot(t_list, np.array([elem[2] for elem in lambda_act_list]), label='Eta_z')  
plt.plot(t_list, np.array([elem[3] for elem in lambda_act_list]), label='Zeta_x')
plt.plot(t_list, np.array([elem[4] for elem in lambda_act_list]), label='Zeta_y')
plt.plot(t_list, np.array([elem[5] for elem in lambda_act_list]), label='Zeta_z')    
plt.xlabel('time[s]',  fontsize=15)
plt.ylabel('[lambda]', fontsize=15)  
plt.title("Tracking Error",  fontsize=20)
plt.ylim([-0.003, 0.003])
#plt.grid(linestyle = 'dotted', linewidth = 1)
plt.grid(True)
plt.legend()

# plt.show()
plt.savefig('Tracking_Error.SVG')