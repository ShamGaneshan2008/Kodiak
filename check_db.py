from sqlalchemy import create_engine, text

engine = create_engine("postgresql://kodiak:kodiak@localhost:5432/kodiak")

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """))

    for row in result.fetchall():
        print(row[0])