create table b1(obj_id integer, ra_core varchar(256), dec_core varchar(256),
                ra_cent varchar(256), dec_cent varchar(256), flux varchar(256),
                core_frac varchar(256), bmaj varchar(256), bmin varchar(256), 
                pa varchar(256), size varchar(10), class varchar(10));

.header ON
.separator ","
.import TrainingSet_B1.csv

create table b2(obj_id integer, ra_core varchar(256), dec_core varchar(256),
                ra_cent varchar(256), dec_cent varchar(256), flux varchar(256),
                core_frac varchar(256), bmaj varchar(256), bmin varchar(256), 
                pa varchar(256), size varchar(10), class varchar(10));

create table b5(obj_id integer, ra_core varchar(256), dec_core varchar(256),
                ra_cent varchar(256), dec_cent varchar(256), flux varchar(256),
                core_frac varchar(256), bmaj varchar(256), bmin varchar(256), 
                pa varchar(256), size varchar(10), class varchar(10));