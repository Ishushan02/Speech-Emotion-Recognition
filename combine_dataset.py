import os


'''
I have in total of 5 Datasets available from different set of Repositories

CREMAD

    CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were 
    from 48 male and 43 female actors between the ages of 20 and 74 coming from a 
    variety of races and ethnicities (African America, Asian, Caucasian, Hispanic,
    and Unspecified). Actors spoke from a selection of 12 sentences. The sentences 
    were presented using one of six different emotions (Anger, Disgust,Fear, Happy, 
    Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

EMODB
RAVDESS
SAVEE
TESS

'''

combined_audio_path = "/Users/ishananand/Desktop/ser/combined_dataset/"

cremad_path = "/Users/ishananand/Desktop/ser/dataset/cremad_dataset/"
cremad_dir = os.listdir(cremad_path)

print("Total Length of CREMAD Audio's: ", len(cremad_dir))

for i in range(len(cremad_dir)):
    file_path = cremad_path + cremad_dir[i]
    # print(file_path)
    # print(cremad_dir[i].split("_"))
    # The third value of list contains the emotion of the .wav file

    file_name = cremad_dir[i].split("_")

    if file_name[2] == 'SAD':
        audio_path = combined_audio_path + "/sad/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")
    elif file_name[2] == 'ANG':
        audio_path = combined_audio_path + "/angry/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")
    elif file_name[2] == 'DIS':
        audio_path = combined_audio_path + "/disgust/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")
    elif file_name[2] == 'FEA':
        audio_path = combined_audio_path + "/fear/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")
    elif file_name[2] == 'HAP':
        audio_path = combined_audio_path + "/happy/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")
    elif file_name[2] == 'NEU':
        audio_path = combined_audio_path + "/neutral/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")
    else:
        audio_path = combined_audio_path + "/unknown/" + cremad_dir[i]
        os.system(f"cp {file_path} {audio_path}")


