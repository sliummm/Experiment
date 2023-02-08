CREATE DATABASE temp_db ;
CREATE TABLE IF NOT EXISTS temp_db.users(
    id INT(10) NOT NULL PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(200) NOT NULL,
    phone VARCHAR(200),
    address VARCHAR(200),
    password VARCHAR(200) NOT NULL
);
INSERT INTO temp_db.users(name,phone,address,password)
VALUES
('Gaurav','123456789','Mumbai,India','pass134'),
('Sakshi','987654321','Chennai,India','pass456');

SELECT * FROM temp_db.users

