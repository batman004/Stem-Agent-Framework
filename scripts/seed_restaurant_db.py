#!/usr/bin/env python3
"""
Seed script — creates a local SQLite database with realistic restaurant data.

Run once before starting the stem agent:
    python -m scripts.seed_restaurant_db

This generates: data/restaurant.db
The database is then referenced in domains/restaurant_ops.yaml so the stem
agent can inspect the real schema during environment_probe.
"""

import sqlite3
from pathlib import Path
from datetime import date, timedelta
import random

DB_PATH = Path("data/restaurant.db")


def seed():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing database at {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # ── Schema ────────────────────────────────────────────────────────────────
    cur.executescript("""
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS categories (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL UNIQUE,
        description TEXT
    );

    CREATE TABLE IF NOT EXISTS menu_items (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        category_id   INTEGER NOT NULL REFERENCES categories(id),
        name          TEXT    NOT NULL,
        description   TEXT,
        price         REAL    NOT NULL,
        cost          REAL    NOT NULL,
        is_available  INTEGER NOT NULL DEFAULT 1,
        created_at    TEXT    NOT NULL DEFAULT (date('now'))
    );

    CREATE TABLE IF NOT EXISTS suppliers (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        name         TEXT NOT NULL,
        contact_name TEXT,
        phone        TEXT,
        email        TEXT,
        lead_days    INTEGER NOT NULL DEFAULT 2
    );

    CREATE TABLE IF NOT EXISTS ingredients (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        supplier_id INTEGER NOT NULL REFERENCES suppliers(id),
        name        TEXT    NOT NULL,
        unit        TEXT    NOT NULL,
        unit_cost   REAL    NOT NULL,
        stock_qty   REAL    NOT NULL DEFAULT 0,
        reorder_qty REAL    NOT NULL,
        min_stock   REAL    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS menu_item_ingredients (
        menu_item_id  INTEGER NOT NULL REFERENCES menu_items(id),
        ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
        quantity      REAL    NOT NULL,
        PRIMARY KEY (menu_item_id, ingredient_id)
    );

    CREATE TABLE IF NOT EXISTS employees (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL,
        role        TEXT    NOT NULL,
        hourly_rate REAL    NOT NULL,
        hired_at    TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS shifts (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL REFERENCES employees(id),
        shift_date  TEXT    NOT NULL,
        start_time  TEXT    NOT NULL,
        end_time    TEXT    NOT NULL,
        hours       REAL    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS orders (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        order_time   TEXT    NOT NULL,
        table_number INTEGER,
        total        REAL    NOT NULL,
        status       TEXT    NOT NULL DEFAULT 'completed'
    );

    CREATE TABLE IF NOT EXISTS order_items (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id     INTEGER NOT NULL REFERENCES orders(id),
        menu_item_id INTEGER NOT NULL REFERENCES menu_items(id),
        quantity     INTEGER NOT NULL,
        unit_price   REAL    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS inventory_logs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ingredient_id INTEGER NOT NULL REFERENCES ingredients(id),
        log_date      TEXT    NOT NULL,
        change_qty    REAL    NOT NULL,
        reason        TEXT    NOT NULL
    );
    """)

    # ── Seed data ─────────────────────────────────────────────────────────────
    # Categories
    categories = [
        ("Starters", "Appetisers and small plates"),
        ("Mains", "Main course dishes"),
        ("Desserts", "Sweet endings"),
        ("Drinks", "Beverages alcoholic and non-alcoholic"),
        ("Specials", "Chef's daily specials"),
    ]
    cur.executemany("INSERT INTO categories (name, description) VALUES (?, ?)", categories)

    # Suppliers
    suppliers = [
        ("FreshFarm Co.",   "Maria Garcia",  "555-0101", "maria@freshfarm.com",   2),
        ("Metro Meats",     "James Wilson",  "555-0102", "james@metromeats.com",  1),
        ("Oceanic Seafood", "Anna Chen",     "555-0103", "anna@oceanic.com",      3),
        ("Bakery Direct",   "Tom Brown",     "555-0104", "tom@bakerydirect.com",  1),
        ("Drinks World",    "Sara Lee",      "555-0105", "sara@drinksworld.com",  2),
    ]
    cur.executemany(
        "INSERT INTO suppliers (name, contact_name, phone, email, lead_days) VALUES (?,?,?,?,?)",
        suppliers
    )

    # Ingredients
    ingredients = [
        (1, "Tomatoes",        "kg",    1.20,  45.0, 20.0, 10.0),
        (1, "Lettuce",         "head",  0.80,  30.0, 15.0,  8.0),
        (1, "Onions",          "kg",    0.60,  25.0, 10.0,  5.0),
        (2, "Chicken breast",  "kg",    6.50,  18.0, 10.0,  5.0),
        (2, "Beef mince",      "kg",    7.20,  12.0,  8.0,  4.0),
        (2, "Bacon",           "kg",    8.00,   6.0,  4.0,  2.0),
        (3, "Salmon fillet",   "kg",   12.00,   8.0,  5.0,  2.0),
        (3, "Prawns",          "kg",   14.00,   4.0,  3.0,  1.5),
        (4, "Burger buns",     "unit",  0.35,  80.0, 40.0, 20.0),
        (4, "Bread rolls",     "unit",  0.25, 100.0, 50.0, 25.0),
        (1, "Garlic",          "kg",    3.00,  10.0,  5.0,  2.0),
        (1, "Basil",           "bunch", 1.50,  12.0,  6.0,  3.0),
        (5, "Cola",            "can",   0.45, 120.0, 60.0, 30.0),
        (5, "Sparkling water", "bottle",0.80,  80.0, 40.0, 20.0),
        (5, "House wine",      "bottle",4.50,  24.0, 12.0,  6.0),
    ]
    cur.executemany(
        "INSERT INTO ingredients (supplier_id, name, unit, unit_cost, stock_qty, reorder_qty, min_stock) VALUES (?,?,?,?,?,?,?)",
        ingredients
    )

    # Menu items (category_id, name, description, price, cost, is_available)
    menu_items = [
        (1, "Bruschetta",          "Toasted bread with tomatoes and basil",        8.50,  2.10, 1),
        (1, "Garlic Prawns",       "Pan-fried prawns in garlic butter",           12.00,  4.80, 1),
        (2, "Grilled Chicken",     "Free-range breast with seasonal vegetables",  18.50,  7.20, 1),
        (2, "Classic Burger",      "Beef patty, bacon, lettuce, tomato",          16.00,  5.90, 1),
        (2, "Salmon Fillet",       "Pan-seared salmon with lemon butter",         22.00,  9.50, 1),
        (2, "Pasta Bolognese",     "Slow-cooked beef ragu with fresh pasta",      15.00,  4.30, 1),
        (3, "Chocolate Lava Cake", "Warm chocolate cake with vanilla ice cream",   9.00,  2.40, 1),
        (3, "Cheesecake",          "New York style with berry compote",            8.50,  2.20, 1),
        (4, "Cola",                "330ml can",                                    3.50,  0.45, 1),
        (4, "Sparkling Water",     "500ml bottle",                                 3.00,  0.80, 1),
        (4, "House Red Wine",      "175ml glass",                                  7.50,  1.80, 1),
        (5, "Chef's Special",      "Changes daily — ask your server",             24.00,  9.00, 1),
    ]
    cur.executemany(
        "INSERT INTO menu_items (category_id, name, description, price, cost, is_available) VALUES (?,?,?,?,?,?)",
        menu_items
    )

    # Employees
    roles = ["Head Chef", "Sous Chef", "Waiter", "Waiter", "Bar Staff", "Kitchen Hand", "Manager"]
    names = ["Alex Turner", "Jamie Patel", "Sam Rivera", "Chris Kim",
             "Jordan Smith", "Taylor Nguyen", "Morgan Davis"]
    rates = [22.0, 18.0, 13.5, 13.5, 14.0, 12.0, 25.0]
    employees = [
        (names[i], roles[i], rates[i], str(date(2022, 1, 1) + timedelta(days=random.randint(0, 600))))
        for i in range(len(names))
    ]
    cur.executemany(
        "INSERT INTO employees (name, role, hourly_rate, hired_at) VALUES (?,?,?,?)",
        employees
    )

    # Shifts — last 14 days
    today = date.today()
    shifts = []
    for emp_id in range(1, 8):
        for day_offset in range(14):
            shift_date = today - timedelta(days=day_offset)
            # 5 days a week
            if shift_date.weekday() < 5:
                start = "09:00" if emp_id <= 2 else "12:00"
                end   = "17:00" if emp_id <= 2 else "22:00"
                hours = 8.0    if emp_id <= 2 else 10.0
                shifts.append((emp_id, str(shift_date), start, end, hours))
    cur.executemany(
        "INSERT INTO shifts (employee_id, shift_date, start_time, end_time, hours) VALUES (?,?,?,?,?)",
        shifts
    )

    # Orders — last 30 days, 20-60 per day
    random.seed(42)
    order_id = 1
    all_order_items = []
    item_prices = {1:8.5, 2:12.0, 3:18.5, 4:16.0, 5:22.0, 6:15.0,
                   7:9.0, 8:8.5, 9:3.5, 10:3.0, 11:7.5, 12:24.0}

    for day_offset in range(30):
        order_date = today - timedelta(days=day_offset)
        daily_orders = random.randint(20, 60)
        for _ in range(daily_orders):
            hour = random.randint(11, 21)
            minute = random.randint(0, 59)
            order_time = f"{order_date} {hour:02d}:{minute:02d}:00"
            table = random.randint(1, 20)
            # 1-4 items per order
            n_items = random.randint(1, 4)
            items = random.choices(list(item_prices.keys()), k=n_items)
            total = sum(item_prices[m] for m in items)
            cur.execute(
                "INSERT INTO orders (order_time, table_number, total, status) VALUES (?,?,?,?)",
                (order_time, table, total, "completed")
            )
            for menu_item_id in items:
                all_order_items.append((order_id, menu_item_id, 1, item_prices[menu_item_id]))
            order_id += 1

    cur.executemany(
        "INSERT INTO order_items (order_id, menu_item_id, quantity, unit_price) VALUES (?,?,?,?)",
        all_order_items
    )

    # Inventory logs — last 7 days
    inventory_logs = []
    for day_offset in range(7):
        log_date = str(today - timedelta(days=day_offset))
        for ing_id in range(1, 13):
            # Consumption
            inventory_logs.append((ing_id, log_date, -random.uniform(1, 5), "daily usage"))
            # Restocking on Mondays
            if (today - timedelta(days=day_offset)).weekday() == 0:
                inventory_logs.append((ing_id, log_date, random.uniform(10, 30), "delivery"))

    cur.executemany(
        "INSERT INTO inventory_logs (ingredient_id, log_date, change_qty, reason) VALUES (?,?,?,?)",
        inventory_logs
    )

    con.commit()
    con.close()

    print(f"\n✅ Database seeded at {DB_PATH.resolve()}")
    print(f"   Tables created: categories, menu_items, suppliers, ingredients,")
    print(f"                   menu_item_ingredients, employees, shifts, orders,")
    print(f"                   order_items, inventory_logs")
    print(f"\n   Usage in domains/restaurant_ops.yaml:")
    print(f'     resources:')
    print(f'       - type: database')
    print(f'         url: "sqlite:///{DB_PATH.resolve()}"')
    print(f'         label: "Restaurant operations database"')


if __name__ == "__main__":
    seed()
