

# Create DB:

postgres Server is somewhere on the System

- bash
```
sudo -i -u postgres
psql
```

- Postgres Terminal
```
CREATE USER mlops_user WITH PASSWORD 'testing321';

CREATE DATABASE raw_db OWNER mlops_user;
CREATE DATABASE cleaned_db OWNER mlops_user;

GRANT ALL PRIVILEGES ON DATABASE raw_db TO mlops_user;
GRANT ALL PRIVILEGES ON DATABASE cleaned_db TO mlops_user;
```

# Create Tables

- DB raw Data
```
psql -h localhost -d raw_db -U mlops_user
```

```
CREATE TABLE IF NOT EXISTS vehicle_data (
    id SERIAL PRIMARY KEY,
    ffid TEXT NOT NULL,
    height FLOAT NOT NULL,
    loaded BOOLEAN NOT NULL,
    onduty BOOLEAN NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    latitude FLOAT NOT NULL,
    longitude FLOAT NOT NULL,
    speed FLOAT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicle_data (
    id SERIAL PRIMARY KEY,
    ffid TEXT NOT NULL,
    height TEXT NOT NULL,
    loaded TEXT NOT NULL,
    onduty TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    latitude TEXT NOT NULL,
    longitude TEXT NOT NULL,
    speed TEXT NOT NULL
);
```
- No Dupicates allowd
```
CREATE UNIQUE INDEX vehicle_data_unique_idx ON vehicle_data ( ffid, height, loaded, onduty, timestamp, latitude, longitude, speed );
```

- get first 5 rows

```
SELECT * FROM vehicle_data LIMIT 5;
```

- delete all Data from a Table
```
DELETE FROM vehicle_data;
```


- DB cleand Data
```
psql -h localhost -d cleaned_db -U mlops_user
```