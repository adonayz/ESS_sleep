This is the README file for unscripted WASHSensory study1b dataset. 

To "featurize" a person's data, their data first needs to be "unpacked" since the data 
sent by the WASHSensory application is zipped. The processing code can be found in 

```
/datastorage/processing_uploaded_data.py
```

To unpack a participant's data, please run the 

```
unpack_user_uploaded_data(<particpantUUID>,"study1b","unpacked_data")
```

for each user that you wuld like to unpack for. After that, run the 

```
getAllUnpackedDataInfo(<particpantUUID>)
```

to update the list of unpacked files so that the program does not perform redundant 
unzipping. To make the feature file for a participant, run the:

```
collect_user_data_and_save_csv_file(<participantUUID>,"unpacked_data","csvs",["BATHROOM","JOGGING","LYING_DOWN","PHONE_IN_BAG","PHONE_IN_HAND","PHONE_IN_POCKET","PHONE_ON_TABLE_-_FACING_DOWN","PHONE_ON_TABLE_-_FACING_UP","RUNNING","SITTING","SLEEPING","STAIRS_-_GOING_DOWN","STAIRS_-_GOING_UP","STANDING","TALKING_ON_PHONE","TYPING","WALKING","EXERCISING"],\
    ['raw_acc',  'proc_gyro', 'location', 'location_quick_features','raw_magnet', 'proc_magnet', 'audio_naive' ,'lf_measurements', 'discrete_measurements'],\
    add_time_of_day_features=True)
```

for that participant. The newly created feature file can be found in:

```
/datastorage/csvs
```

