import torch
import torch.nn.functional as F

def torch_glm_translate(m,v):
    mb, mt, ms1, ms2 = m.shape
    vb, vt, vs = v.shape
    assert mb == vb, mt == vt
    assert ms1 == ms2 == 4, vs == 3
    device = m.device
    res = torch.zeros_like(m).to(device)
    res[:,:,0:3] = m[:,:,0:3]
    vr = v.reshape(vb, vt, 1, vs)
    # print(torch.matmul(vr,m[:,:,:3]).shape)
    res[:,:,3] = torch.matmul(vr,m[:,:,:3]).reshape(mb, mt,ms2) + m[:,:,3]
    return res

def torch_glm_rotate(m, angle, v):
    mb, mt, ms1, ms2 = m.shape
    anb, ant = angle.shape
    vb, vt, vs = v.shape
    assert mb == vb == anb, mt == vt == ant
    assert ms1 == ms2 == 4, vs == 3
    device = m.device
    c = torch.cos(angle)
    s = torch.sin(angle)
    axis = F.normalize(v.clone(),p=2,dim=-1)
    temp = (1-c).reshape(anb, ant, 1)*axis

    rot = torch.zeros_like(m).to(device)
    res = torch.zeros_like(m).to(device)
    rot[:,:,0,0] = c + temp[:,:,0] * axis[:,:,0]
    rot[:,:,0,1] = temp[:,:,0] * axis[:,:,1] + s*axis[:,:,2]
    rot[:,:,0,2] = temp[:,:,0] * axis[:,:,2] - s*axis[:,:,1]
    rot[:,:,1,0] = temp[:,:,1] * axis[:,:,0] - s * axis[:,:,2]
    rot[:,:,1,1] = c + temp[:,:,1] * axis[:,:,1]
    rot[:,:,1,2] = temp[:,:,1] * axis[:,:,2] + s * axis[:,:,0]

    rot[:,:,2,0] = temp[:,:,2] * axis[:,:,0] + s * axis[:,:,1]
    rot[:,:,2,1] = temp[:,:,2] * axis[:,:,1] - s * axis[:,:,0]
    rot[:,:,2,2] = c + temp[:,:,2] * axis[:,:,2]

    # res[:,:,0] = m[:,:,:,0] * rot[:,:,0:1,0] + m[:,:,:,1] * rot[:,:,0:1,1] + m[:,:,:,2] * rot[:,:,0:1,2]
    # res[:,:,1] = m[:,:,:,0] * rot[:,:,1:2,0] + m[:,:,:,1] * rot[:,:,1:2,1] + m[:,:,:,2] * rot[:,:,1:2,2]
    # res[:,:,2] = m[:,:,:,0] * rot[:,:,2:3,0] + m[:,:,:,1] * rot[:,:,2:3,1] + m[:,:,:,2] * rot[:,:,2:3,2]
    res[:,:,0:3] = torch.matmul(rot[:,:,0:3,0:3], m[:,:,0:3])
    res[:,:,3] = m[:,:,3]
    return res
    
