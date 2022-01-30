--  Write a SQL script that creates a trigger that decreases the quantity of an item after adding a new order.
--  Quantity in the table items can be negative.
CREATE TRIGGER decreases AFTER INSERT ON orders
FOR EACH ROW
  UPDATE items
  SET items.quantity = items.quantity - NEW.number
  WHERE items.name = NEW.item_name;
