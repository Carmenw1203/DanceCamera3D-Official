__all__ = ['Vmd']
from functools import reduce
import struct
import json
from tqdm import tqdm
from my_utils.mmd_skeleton import Mmd_Skeleton

def pad_zero(b_array,pad_l):
    # for pi in range(pad_l):
    b_array += bytes([0]*pad_l)
    return b_array

def pad_zero_list(b_array_list,pad_l):
    # for pi in range(pad_l):
    b_array_list.append(bytes([0]*pad_l))
    return b_array_list

class Vmd:

    def __init__(self):
        pass

    @staticmethod
    def from_file(filename, model_name_encode="shift-JIS"):

        with open(filename, "rb") as f:
            array = b''.join(list(f))
        # print(len(array))
        
        vmd = Vmd()

        VersionInformation = array[:30].decode("ascii")
        if VersionInformation.startswith("Vocaloid Motion Data file"):
            vision = 1
        elif VersionInformation.startswith("Vocaloid Motion Data 0002"):
            vision = 2
        else:
            raise Exception("unknow vision")

        vmd.vision = vision

        vmd.model_name = array[30: 30+10*vision].split(bytes([0]))[0].decode(model_name_encode,errors='ignore')
        vmd.bone_keyframe_number = int.from_bytes(array[30+10*vision: 30+10*vision+4], byteorder='little', signed=False)
        vmd.bone_keyframe_record = []
        vmd.morph_keyframe_record = []
        vmd.camera_keyframe_record = []
        vmd.light_keyframe_record = []

        current_index = 34+10 * vision
        
        for i in range(vmd.bone_keyframe_number):
            vmd.bone_keyframe_record.append({
                "BoneName": array[current_index: current_index+15].split(bytes([0]))[0].decode(model_name_encode,errors='ignore'),
                "FrameTime": struct.unpack("<I", array[current_index+15: current_index+19])[0],
                "Position": {"x": struct.unpack("<f", array[current_index+19: current_index+23])[0],
                            "y": struct.unpack("<f", array[current_index+23: current_index+27])[0],
                            "z": struct.unpack("<f", array[current_index+27: current_index+31])[0]
                            },
                "Rotation":{"x": struct.unpack("<f", array[current_index+31: current_index+35])[0],
                            "y": struct.unpack("<f", array[current_index+35: current_index+39])[0],
                            "z": struct.unpack("<f", array[current_index+39: current_index+43])[0],
                            "w": struct.unpack("<f", array[current_index+43: current_index+47])[0]
                            },
                "Curve":{
                    "x":(array[current_index+47], array[current_index+51], array[current_index+55], array[current_index+59]),
                    "y":(array[current_index+63], array[current_index+67], array[current_index+71], array[current_index+75]),
                    "z":(array[current_index+79], array[current_index+83], array[current_index+87], array[current_index+91]),
                    "r":(array[current_index+95], array[current_index+99], array[current_index+103], array[current_index+107])
                }
            

            })
            
            current_index += 111
        
        vmd.morph_keyframe_number = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        current_index += 4

        for i in range(vmd.morph_keyframe_number):
            vmd.morph_keyframe_record.append({
                'MorphName': array[current_index: current_index+15].split(bytes([0]))[0].decode(model_name_encode,errors='ignore'),
                'FrameTime': struct.unpack("<I", array[current_index+15: current_index+19])[0],
                'Weight': struct.unpack("<f", array[current_index+19: current_index+23])[0]
            })
            current_index += 23

        vmd.camera_keyframe_number = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        current_index += 4

        for i in range(vmd.camera_keyframe_number):
            vmd.camera_keyframe_record.append({
                'FrameTime': struct.unpack("<I", array[current_index: current_index+4])[0],
                'Distance': struct.unpack("<f", array[current_index+4: current_index+8])[0],
                "Position": {"x": struct.unpack("<f", array[current_index+8: current_index+12])[0],
                            "y": struct.unpack("<f", array[current_index+12: current_index+16])[0],
                            "z": struct.unpack("<f", array[current_index+16: current_index+20])[0]
                            },
                "Rotation":{"x": struct.unpack("<f", array[current_index+20: current_index+24])[0],
                            "y": struct.unpack("<f", array[current_index+24: current_index+28])[0],
                            "z": struct.unpack("<f", array[current_index+28: current_index+32])[0]
                            },
                "Curve": tuple(b for b in array[current_index+32: current_index+56]),
                "ViewAngle": struct.unpack("<I", array[current_index+56: current_index+60])[0],
                "Orthographic": array[60]
            })
            current_index += 61

        vmd.light_keyframe_number = int.from_bytes(array[current_index: current_index+4], byteorder="little", signed=False)
        current_index += 4

        for i in range(vmd.light_keyframe_number):
            vmd.light_keyframe_record.append({
                'FrameTime': struct.unpack("<I", array[current_index: current_index+4])[0],
                'Color': {
                    'r': struct.unpack("<f", array[current_index+4: current_index+8])[0],
                    'g': struct.unpack("<f", array[current_index+8: current_index+12])[0],
                    'b': struct.unpack("<f", array[current_index+12: current_index+16])[0]
                },
                'Direction':{"x": struct.unpack("<f", array[current_index+16: current_index+20])[0],
                            "y": struct.unpack("<f", array[current_index+20: current_index+24])[0],
                            "z": struct.unpack("<f", array[current_index+24: current_index+28])[0]
                            }
            })
            current_index += 28
        # print(current_index)
        vmd_dict = {}
        vmd_dict['Vision'] = vision
        vmd_dict['ModelName'] = vmd.model_name
        vmd_dict['BoneKeyFrameNumber'] = vmd.bone_keyframe_number
        vmd_dict['BoneKeyFrameRecord'] = vmd.bone_keyframe_record
        vmd_dict['MorphKeyFrameNumber'] = vmd.morph_keyframe_number
        vmd_dict['MorphKeyFrameRecord'] = vmd.morph_keyframe_record
        vmd_dict['CameraKeyFrameNumber'] = vmd.camera_keyframe_number
        vmd_dict['CameraKeyFrameRecord'] = vmd.camera_keyframe_record
        vmd_dict['LightKeyFrameNumber'] = vmd.light_keyframe_number
        vmd_dict['LightKeyFrameRecord'] = vmd.light_keyframe_record

        vmd.dict = vmd_dict

        return vmd
    def file_compare(filename1,filename2):
        with open(filename1, "rb") as f1:
            array1 = bytes(reduce(lambda x, y: x+y, list(f1)))
        with open(filename2, "rb") as f2:
            array2 = bytes(reduce(lambda x, y: x+y, list(f2)))
        if array1 == array2:
            print(filename1+" and "+filename2+" are the same")
        else:
            print(filename1+" and "+filename2+" are different")
    
    def fuse_motion_camera(motion_file_name, camera_file_name, fuse_file_name, model_name_encode="shift-JIS"):
        with open (motion_file_name, 'r') as mf:
            motion_data = json.load(mf)
        with open (camera_file_name, 'r') as cf:
            camera_data = json.load(cf)
        
        array_list = []
        vision = 2
        VersionInformation = "Vocaloid Motion Data 0002"
        VersionInformation = VersionInformation.encode("ascii")
        array_list.append(VersionInformation)
        array_list = pad_zero_list(array_list,30 - len(VersionInformation))

        model_name = motion_data["ModelName"]#这个无所谓
        model_name = model_name.encode(model_name_encode,errors='ignore')
        array_list.append(model_name)
        array_list = pad_zero_list(array_list,10*vision - len(model_name))
        
        bone_keyframe_number = motion_data["BoneKeyFrameNumber"]
        keyframe_array_list = []
        valid_frame_number = 0
        for i in tqdm(range(bone_keyframe_number), desc='Processing'):
            if motion_data["BoneKeyFrameRecord"][i]["BoneName"] not in Mmd_Skeleton.standard_bones:
                # print(motion_data["BoneKeyFrameRecord"][i]["BoneName"])
                continue
            # print(motion_data["BoneKeyFrameRecord"][i]["BoneName"])
            valid_frame_number += 1
            bone_name = motion_data["BoneKeyFrameRecord"][i]["BoneName"].encode(model_name_encode,errors='ignore')
            
            keyframe_array_list.append(bone_name)
            keyframe_array_list = pad_zero_list(keyframe_array_list,15 - len(bone_name))
            keyframe_array_list.append(struct.pack("<Ifffffff",motion_data["BoneKeyFrameRecord"][i]["FrameTime"],
                                            motion_data["BoneKeyFrameRecord"][i]["Position"]["x"],
                                            motion_data["BoneKeyFrameRecord"][i]["Position"]["y"],
                                            motion_data["BoneKeyFrameRecord"][i]["Position"]["z"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["x"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["y"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["z"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["w"]))
            
            
            keyframe_array_list = pad_zero_list(keyframe_array_list,4*4*4)
            
        array_list.append(valid_frame_number.to_bytes(4,byteorder='little', signed=False))
        array_list = array_list + keyframe_array_list
        

        
        morph_keyframe_number = 0
        morph_keyframe_array_list = []
        array_list.append(morph_keyframe_number.to_bytes(4,byteorder='little', signed=False))
        for i in range(morph_keyframe_number):
            morph_name = motion_data["MorphKeyFrameRecord"][i]["MorphName"].encode(model_name_encode,errors='ignore')
            morph_keyframe_array_list.append(morph_name)
            morph_keyframe_array_list = pad_zero(array,15 - len(morph_name))
            morph_keyframe_array_list.append(struct.pack("<I",motion_data["MorphKeyFrameRecord"][i]["FrameTime"]))
            morph_keyframe_array_list.append(struct.pack("<f",motion_data["MorphKeyFrameRecord"][i]["Weight"]))
        
        array_list = array_list + morph_keyframe_array_list


        camera_keyframe_number = camera_data["CameraKeyFrameNumber"]
        array_list.append(camera_keyframe_number.to_bytes(4,byteorder='little', signed=False))
        for i in range(camera_keyframe_number):
            array_list.append(struct.pack("<Ifffffff",
            camera_data["CameraKeyFrameRecord"][i]["FrameTime"],
            camera_data["CameraKeyFrameRecord"][i]["Distance"],
            camera_data["CameraKeyFrameRecord"][i]["Position"]["x"],
            camera_data["CameraKeyFrameRecord"][i]["Position"]["y"],
            camera_data["CameraKeyFrameRecord"][i]["Position"]["z"],
            camera_data["CameraKeyFrameRecord"][i]["Rotation"]["x"],
            camera_data["CameraKeyFrameRecord"][i]["Rotation"]["y"],
            camera_data["CameraKeyFrameRecord"][i]["Rotation"]["z"]))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][0].to_bytes(1,byteorder='little', signed=False))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][1].to_bytes(1,byteorder='little', signed=False))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][2].to_bytes(1,byteorder='little', signed=False))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][3].to_bytes(1,byteorder='little', signed=False))
            array_list = pad_zero_list(array_list,20)
            array_list.append(struct.pack("<f",camera_data["CameraKeyFrameRecord"][i]["ViewAngle"]))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Orthographic"].to_bytes(1,byteorder='little', signed=False))

        light_keyframe_number = 0
        array_list.append(light_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        for i in range(light_keyframe_number):
            array_list.append(struct.pack("<Iffffff",
            motion_data["LightKeyFrameRecord"][i]["FrameTime"],
            motion_data["LightKeyFrameRecord"][i]["Color"]["r"],
            motion_data["LightKeyFrameRecord"][i]["Color"]["g"],
            motion_data["LightKeyFrameRecord"][i]["Color"]["b"],
            motion_data["LightKeyFrameRecord"][i]["Direction"]["x"],
            motion_data["LightKeyFrameRecord"][i]["Direction"]["y"],
            motion_data["LightKeyFrameRecord"][i]["Direction"]["z"]))

        array = b''.join(array_list)
        with open(fuse_file_name, "wb") as ff:
            ff.write(array)

    def saba_motion_json_to_vmd(motion_json, motion_vmd, model_name_encode="shift-JIS"):
        with open (motion_json, 'r') as mf:
            motion_data = json.load(mf)
        
        array_list = [] 
        vision = 2
        VersionInformation = "Vocaloid Motion Data 0002"
        VersionInformation = VersionInformation.encode("ascii")
        array_list.append(VersionInformation)
        array_list = pad_zero_list(array_list,30 - len(VersionInformation))

        model_name = "0" 
        model_name = model_name.encode(model_name_encode,errors='ignore')
        array_list.append(model_name)
        array_list = pad_zero_list(array_list,10*vision - len(model_name))
        
        bone_keyframe_number = motion_data["BoneKeyFrameNumber"]
        # bone_keyframe_number = 50000 #test debug
        # array += bone_keyframe_number.to_bytes(4,byteorder='little', signed=False)
        # print("fuse motion info……")
        keyframe_array_list = []
        valid_frame_number = 0
        for i in tqdm(range(bone_keyframe_number), desc='Processing'):
            if motion_data["BoneKeyFrameRecord"][i]["BoneName"] not in Mmd_Skeleton.standard_bones:
                # print(motion_data["BoneKeyFrameRecord"][i]["BoneName"])
                continue
            # print(motion_data["BoneKeyFrameRecord"][i]["BoneName"])
            valid_frame_number += 1
            bone_name = motion_data["BoneKeyFrameRecord"][i]["BoneName"].encode(model_name_encode,errors='ignore')
            
            keyframe_array_list.append(bone_name)
            keyframe_array_list = pad_zero_list(keyframe_array_list,15 - len(bone_name))
            keyframe_array_list.append(struct.pack("<Ifffffff",motion_data["BoneKeyFrameRecord"][i]["FrameTime"],
                                            motion_data["BoneKeyFrameRecord"][i]["Position"]["x"],
                                            motion_data["BoneKeyFrameRecord"][i]["Position"]["y"],
                                            motion_data["BoneKeyFrameRecord"][i]["Position"]["z"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["x"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["y"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["z"],
                                            motion_data["BoneKeyFrameRecord"][i]["Rotation"]["w"]))
            
            keyframe_array_list = pad_zero_list(keyframe_array_list,4*4*4)
            
        array_list.append(valid_frame_number.to_bytes(4,byteorder='little', signed=False))
        array_list = array_list + keyframe_array_list

        morph_keyframe_number = 0
        array_list.append(morph_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        camera_keyframe_number = 0
        array_list.append(camera_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        light_keyframe_number = 0
        array_list.append(light_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        array = b''.join(array_list)
        with open(motion_vmd, "wb") as ff:
            ff.write(array)

    def saba_camera_json_to_vmd(camera_json, camera_vmd, model_name_encode="shift-JIS"):
        with open (camera_json, 'r') as cf:
            camera_data = json.load(cf)
        
        array_list = [] 
        vision = 2
        VersionInformation = "Vocaloid Motion Data 0002"
        VersionInformation = VersionInformation.encode("ascii")
        array_list.append(VersionInformation)
        array_list = pad_zero_list(array_list,30 - len(VersionInformation))

        model_name = "0" 
        model_name = model_name.encode(model_name_encode,errors='ignore')
        array_list.append(model_name)
        array_list = pad_zero_list(array_list,10*vision - len(model_name))
        
        bone_keyframe_number = 0
        array_list.append(bone_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        morph_keyframe_number = 0
        array_list.append(morph_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        camera_keyframe_number = camera_data["CameraKeyFrameNumber"]
        array_list.append(camera_keyframe_number.to_bytes(4,byteorder='little', signed=False))
        for i in range(camera_keyframe_number):
            array_list.append(struct.pack("<Ifffffff",
            camera_data["CameraKeyFrameRecord"][i]["FrameTime"],
            float(camera_data["CameraKeyFrameRecord"][i]["Distance"]),
            float(camera_data["CameraKeyFrameRecord"][i]["Position"]["x"]),
            float(camera_data["CameraKeyFrameRecord"][i]["Position"]["y"]),
            float(camera_data["CameraKeyFrameRecord"][i]["Position"]["z"]),
            float(camera_data["CameraKeyFrameRecord"][i]["Rotation"]["x"]),
            float(camera_data["CameraKeyFrameRecord"][i]["Rotation"]["y"]),
            float(camera_data["CameraKeyFrameRecord"][i]["Rotation"]["z"])))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][0].to_bytes(1,byteorder='little', signed=False))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][1].to_bytes(1,byteorder='little', signed=False))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][2].to_bytes(1,byteorder='little', signed=False))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Curve"][3].to_bytes(1,byteorder='little', signed=False))
            array_list = pad_zero_list(array_list,20)
            array_list.append(struct.pack("<f",camera_data["CameraKeyFrameRecord"][i]["ViewAngle"]))
            array_list.append(camera_data["CameraKeyFrameRecord"][i]["Orthographic"].to_bytes(1,byteorder='little', signed=False))

        light_keyframe_number = 0
        array_list.append(light_keyframe_number.to_bytes(4,byteorder='little', signed=False))

        array = b''.join(array_list)
        with open(camera_vmd, "wb") as ff:
            ff.write(array)