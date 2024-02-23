import torch

teacher_weights_path = './weights/teacher_weights.pth'
student_weights_path = 'ckpt/sam_med3d_turbo.pth'

teacher_state_dict = torch.load(teacher_weights_path)
student_state_dict = torch.load(student_weights_path)

def compare_weights(teacher_state_dict, student_state_dict):
    diff = 0
    matched_params = 0
    for key in teacher_state_dict:
        if key in student_state_dict:
            # 计算L2范数差异
            diff += torch.norm(teacher_state_dict[key] - student_state_dict[key], 2).item()
            matched_params += 1
    return diff, matched_params

# 计算并打印权重差异
#weight_diff, matched_params = compare_weights(teacher_state_dict, student_state_dict)
#print(f"Weight difference between teacher and student model: {weight_diff}")
#print(f"Number of matched parameters: {matched_params}")
