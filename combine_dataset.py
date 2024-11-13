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

    The EMODB database is the freely available German emotional database. The database
    is created by the Institute of Communication Science, Technical University, Berlin,
    Germany. Ten professional speakers (five males and five females) participated in data
    recording. The database contains a total of 535 utterances. The EMODB database comprises
    of seven emotions: 1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral. 
    The data was recorded at a 48-kHz sampling rate and then down-sampled to 16-kHz.

    letter 	emotion     (german)	
        A	anger	        W	    
        B	boredom	        L	    
        D	disgust	        E	    
        F	anxiety/fear	A	    
        H	happiness	    F	    
        S	sadness	        T	    


RAVDESS
SAVEE
TESS

'''

combined_audio_path = "/Users/ishananand/Desktop/ser/combined_dataset/"





def combineCREMAD(cremad_path, combined_audio_path):
    cremad_dir = os.listdir(cremad_path)
    # print("Total Length of CREMAD Audio's: ", len(cremad_dir))

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


def combineEMODB(emodbpath, combined_audio_path):
    emodbdir = os.listdir(emodbpath)

    for i in range(len(emodbdir)):
        file_path = emodbpath + emodbdir[i]
        # print(file_path)
        # print(cremad_dir[i].split("_"))
        # The third value of list contains the emotion of the .wav file

        file_name = emodbdir[i]

        if file_name[5] == 'T':
            audio_path = combined_audio_path + "/sad/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif file_name[5] == 'W':
            audio_path = combined_audio_path + "/angry/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif file_name[5] == 'L':
            audio_path = combined_audio_path + "/bored/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif file_name[5] == 'E':
            audio_path = combined_audio_path + "/disgust/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif file_name[5] == 'A':
            audio_path = combined_audio_path + "/fear/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif file_name[5] == 'F':
            audio_path = combined_audio_path + "/happy/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif file_name[5] == 'N':
            audio_path = combined_audio_path + "/neutral/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")
        else:
            audio_path = combined_audio_path + "/unknown/" + emodbdir[i]
            os.system(f"cp {file_path} {audio_path}")




if __name__ == "__main__":

    cremad_path = "/Users/ishananand/Desktop/ser/dataset/cremad_dataset/"
    # combineCREMAD(cremad_path, combined_audio_path)
    print("All CREMAD data is completed formatted and stored in there respective Folders")

    emodb_path = "/Users/ishananand/Desktop/ser/dataset/emodb_dataset/"
    combineEMODB(emodb_path, combined_audio_path)
    print("All EMODB data is completed formatted and stored in there respective Folders")

