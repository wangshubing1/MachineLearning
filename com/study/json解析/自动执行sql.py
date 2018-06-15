import phoenixdb
import phoenixdb.cursor

database_url = 'http://172.28.3.193:8765/'
conn = phoenixdb.connect(database_url, autocommit=True)

cursor = conn.cursor()
# cursor.execute("CREATE TABLE users2 (id INTEGER PRIMARY KEY, username VARCHAR)")
# cursor.execute("UPSERT INTO users2 VALUES (?, ?)", (1, 'admin'))
cursor.execute('''select * from "user.t_user" limit 10''')
for a in cursor.fetchall():
    print(a)
# print(cursor.fetchone())
# print (cursor.fetchall())

# cursor = conn.cursor(cursor_factory=phoenixdb.cursor.DictCursor)
# cursor.execute("SELECT * FROM users2 WHERE id=1")
# print (cursor.fetchone()['USERNAME'])