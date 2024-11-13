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

    This portion of the RAVDESS contains 1440 files: 60 trials per actor x 24 actors = 1440. 
    The RAVDESS contains 24 professional actors (12 female, 12 male), vocalizing two lexically-matched
    statements in a neutral North American accent. Speech emotions includes calm, happy, sad,
    angry, fearful, surprise, and disgust expressions. Each expression is produced at two levels
    of emotional intensity (normal, strong), with an additional neutral expression.

    Filename identifiers
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). Note: There is no strong intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).



SAVEE
TESS

'''

combined_audio_path = "/Users/ishananand/Desktop/ser/combined_dataset/"





def combineCREMAD(cremad_path, combined_audio_path):
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

# extra bored
def combineEMODB(emodbpath, combined_audio_path):
    emodbdir = os.listdir(emodbpath)

    print("Total Length of EMODB Audio's: ", len(emodbdir))
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


# extra calm, surprised
def combineRAVEDESS(ravedesspath, combined_audio_path):

    ravedessactordir = os.listdir(ravedesspath)

    totalaudios = 0
    for each_actor in ravedessactordir:
        audios = os.listdir(ravedesspath + each_actor)
        totalaudios += len(audios)
        for i in range(len(audios)):
            

            file_name = audios[i].split('.')[0].split('-')
            file_path = ravedesspath + each_actor + "/"  + audios[i]

            if file_name[2] == '04':
                audio_path = combined_audio_path + "/sad/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '05':
                audio_path = combined_audio_path + "/angry/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            # elif file_name[2] == 'L':
            #     audio_path = combined_audio_path + "/bored/" + audios[i]
            #     os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '07':
                audio_path = combined_audio_path + "/disgust/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '06':
                audio_path = combined_audio_path + "/fear/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '03':
                audio_path = combined_audio_path + "/happy/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '01':
                audio_path = combined_audio_path + "/neutral/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '02':
                audio_path = combined_audio_path + "/calm/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            elif file_name[2] == '08':
                audio_path = combined_audio_path + "/surprise/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")
            else:
                audio_path = combined_audio_path + "/unknown/" + audios[i]
                os.system(f"cp {file_path} {audio_path}")

    print("Total Length of RAVEDESS Audio's: ", totalaudios)


def combineSAVEE(savee_path, combined_audio_path):
    savee_dir = os.listdir(savee_path)
    print("Total Length of SAVEE Audio's: ", len(savee_dir))

    for i in range(len(savee_dir)):
        file_path = savee_path + savee_dir[i]
        # print(file_path)
        # print(cremad_dir[i].split("_"))
        # The third value of list contains the emotion of the .wav file

        file_name = savee_dir[i].split("_")[1]
        word = file_name[:-6]

        if word == 'sa':
            audio_path = combined_audio_path + "/sad/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif word == 'a':
            audio_path = combined_audio_path + "/angry/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif word == 'd':
            audio_path = combined_audio_path + "/disgust/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif word == 'f':
            audio_path = combined_audio_path + "/fear/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif word == 'h':
            audio_path = combined_audio_path + "/happy/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif word == 'n':
            audio_path = combined_audio_path + "/neutral/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")
        elif word == 'su':
                audio_path = combined_audio_path + "/surprise/" + savee_dir[i]
                os.system(f"cp {file_path} {audio_path}")
        else:
            audio_path = combined_audio_path + "/unknown/" + savee_dir[i]
            os.system(f"cp {file_path} {audio_path}")




if __name__ == "__main__":

    cremad_path = "/Users/ishananand/Desktop/ser/dataset/cremad_dataset/"
    # combineCREMAD(cremad_path, combined_audio_path)
    print("All CREMAD data is completed formatted and stored in there respective Folders")

    emodb_path = "/Users/ishananand/Desktop/ser/dataset/emodb_dataset/"
    # combineEMODB(emodb_path, combined_audio_path)
    print("All EMODB data is completed formatted and stored in there respective Folders")

    ravedess_path = "/Users/ishananand/Desktop/ser/dataset/ravdess_dataset/"
    # combineRAVEDESS(ravedess_path, combined_audio_path)
    print("All RAVEDESS data is completed formatted and stored in there respective Folders")


    savee_path = "/Users/ishananand/Desktop/ser/dataset/savee_dataset/"
    combineSAVEE(savee_path, combined_audio_path)
    print("All SAVEE data is completed formatted and stored in there respective Folders")




