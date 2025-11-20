# /insert_data

curl -X POST "http://127.0.0.1:8000/insert_data"   -H "accept: application/json"   -H "Content-Type: multipart/form-data"   -F "file=@test.csv"

test.csv: 
FFID,Height,Beladen,OnDuty,TimeStamp,Latitude,Longitude,Speed
F123,2.5,True,False,2025-11-19T12:34:56,48.123,11.567,60.5
F124,3.0,False,True,2025-11-19T12:35:00,48.124,11.568,55.0
F125,2.8,True,True,2025-11-19T12:36:00,48.125,11.569,58.0

- /ping
curl -X GET "http://127.0.0.1:8000/ping"

- /test
curl -X GET "http://127.0.0.1:8000/test?test=HELLOWORLD"
?test  ---> Key
=HelloWorld ---> Value