--Down Script
-- Drop Triggers if they exist
IF OBJECT_ID('trg_calculate_quote_details_total', 'TR') IS NOT NULL
    DROP TRIGGER trg_calculate_quote_details_total;
    
IF OBJECT_ID('trg_calculate_invoice_details_total', 'TR') IS NOT NULL
    DROP TRIGGER trg_calculate_invoice_details_total;

-- Drop Foreign Key Constraints if they exist
IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_payments_invoice_id')
    ALTER TABLE Payments DROP CONSTRAINT FK_payments_invoice_id;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_invoice_details_invoice_id')
    ALTER TABLE invoice_details DROP CONSTRAINT FK_invoice_details_invoice_id;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_invoice_details_product_id')
    ALTER TABLE invoice_details DROP CONSTRAINT FK_invoice_details_product_id;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_invoice_quote_id_version')
    ALTER TABLE invoice DROP CONSTRAINT FK_invoice_quote_id_version;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_quote_details_quote_id_version')
    ALTER TABLE quote_details DROP CONSTRAINT FK_quote_details_quote_id_version;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_quote_details_product_id')
    ALTER TABLE quote_details DROP CONSTRAINT FK_quote_details_product_id;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_quote_discounts_discount_id')
    ALTER TABLE quote_discounts DROP CONSTRAINT FK_quote_discounts_discount_id;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_quote_discounts_quote_id_version')
    ALTER TABLE quote_discounts DROP CONSTRAINT FK_quote_discounts_quote_id_version;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_quote_customer_id')
    ALTER TABLE quote DROP CONSTRAINT FK_quote_customer_id;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'FK_quote_delivery_state')
    ALTER TABLE quote DROP CONSTRAINT FK_quote_delivery_state;

IF EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS 
           WHERE CONSTRAINT_NAME = 'UQ_customers_email')
    ALTER TABLE customers DROP CONSTRAINT UQ_customers_email;

-- Drop Default Constraints if they exist
IF EXISTS (SELECT * FROM sys.default_constraints WHERE object_id = OBJECT_ID('DF_quote_estimate_status'))
    ALTER TABLE quote DROP CONSTRAINT DF_quote_estimate_status;

IF EXISTS (SELECT * FROM sys.default_constraints WHERE object_id = OBJECT_ID('DF_quote_approved_by_customer'))
    ALTER TABLE quote DROP CONSTRAINT DF_quote_approved_by_customer;

-- Drop Tables if they exist
DROP TABLE IF EXISTS Payments;
DROP TABLE IF EXISTS invoice_details;
DROP TABLE IF EXISTS invoice;
DROP TABLE IF EXISTS quote_details;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS quote_discounts;
DROP TABLE IF EXISTS quote;
DROP TABLE IF EXISTS discounts;
DROP TABLE IF EXISTS sales_tax;
DROP TABLE IF EXISTS customers;

-- Drop Stored Procedures if they exist
IF OBJECT_ID('GenerateQuote', 'P') IS NOT NULL
    DROP PROCEDURE GenerateQuote;

IF OBJECT_ID('AddItemsToQuoteDetails', 'P') IS NOT NULL
    DROP PROCEDURE AddItemsToQuoteDetails;

IF OBJECT_ID('ProcessPayment', 'P') IS NOT NULL
    DROP PROCEDURE ProcessPayment;

IF OBJECT_ID('InsertCustomer', 'P') IS NOT NULL
    DROP PROCEDURE InsertCustomer;

IF OBJECT_ID('InsertOrUpdateDiscount', 'P') IS NOT NULL
    DROP PROCEDURE InsertOrUpdateDiscount;

IF OBJECT_ID('InsertOrUpdateProduct', 'P') IS NOT NULL
    DROP PROCEDURE InsertOrUpdateProduct;

IF OBJECT_ID('GenerateInvoiceFromQuote', 'P') IS NOT NULL
    DROP PROCEDURE GenerateInvoiceFromQuote;

IF OBJECT_ID('AddOrUpdateInvoiceDetails', 'P') IS NOT NULL
    DROP PROCEDURE AddOrUpdateInvoiceDetails;

GO

-- Up Script

-- Create Customers Table
CREATE TABLE customers (
    customers_customer_id INT IDENTITY(1,1) CONSTRAINT PK_customers_customer_id PRIMARY KEY,
    customers_first_name VARCHAR(50),
    customers_last_name VARCHAR(50),
    customers_billing_address VARCHAR(255),
    customers_billing_city VARCHAR(50),
    customers_billing_state VARCHAR(50),
    customers_billing_zip VARCHAR(20),
    customers_email VARCHAR(100),
    customers_phone_number VARCHAR(20),
    CONSTRAINT UQ_customers_email UNIQUE (customers_email),
);
GO
-- Create Sales Tax Table
CREATE TABLE sales_tax (
    sales_tax_state VARCHAR(50) CONSTRAINT PK_sales_tax_state PRIMARY KEY,
    sales_tax_tax_rate DECIMAL(5, 2)
);
GO
-- Create Discounts Table
CREATE TABLE discounts (
    discounts_discount_id INT IDENTITY(1,1) CONSTRAINT PK_discounts_discount_id PRIMARY KEY,
    discounts_name VARCHAR(100),
    discounts_discount_value DECIMAL(5, 2)
);
GO
CREATE TABLE quote (
    quote_quote_id INT,
    quote_version INT,
    quote_customer_id INT,
    quote_date_updated DATE,
    quote_delivery_street VARCHAR(255),
    quote_delivery_city VARCHAR(50),
    quote_delivery_state VARCHAR(50),
    quote_delivery_zip VARCHAR(20),
    quote_delivery_date DATE,
    quote_estimate_status VARCHAR(50) CONSTRAINT DF_quote_estimate_status DEFAULT 'No',  
    quote_proposal TEXT,
    quote_expiration DATE,
    quote_approved_by_customer VARCHAR(3) CONSTRAINT DF_quote_approved_by_customer DEFAULT 'No',  
    CONSTRAINT PK_quote_quote_id_version PRIMARY KEY (quote_quote_id, quote_version),
    CONSTRAINT FK_quote_customer_id FOREIGN KEY (quote_customer_id) REFERENCES customers(customers_customer_id),
    CONSTRAINT FK_quote_delivery_state FOREIGN KEY (quote_delivery_state) REFERENCES sales_tax(sales_tax_state)
);
GO
-- Create Quote Discounts Table
CREATE TABLE quote_discounts (
    quote_discounts_discount_id INT,
    quote_discounts_quote_id INT,
    quote_discounts_version INT,
    CONSTRAINT PK_quote_discounts_discount_id_quote_id_version PRIMARY KEY (quote_discounts_discount_id, quote_discounts_quote_id, quote_discounts_version),
    CONSTRAINT FK_quote_discounts_discount_id FOREIGN KEY (quote_discounts_discount_id) REFERENCES discounts(discounts_discount_id),
    CONSTRAINT FK_quote_discounts_quote_id_version FOREIGN KEY (quote_discounts_quote_id, quote_discounts_version) REFERENCES quote(quote_quote_id, quote_version)
);
GO
-- Create Products Table
CREATE TABLE products (
    products_product_id INT IDENTITY(1,1) CONSTRAINT PK_products_product_id PRIMARY KEY,
    products_name VARCHAR(100),
    products_description TEXT,
    products_price DECIMAL(10, 2),
    products_unit_of_measure VARCHAR(50)
);
GO
-- Create Quote Details Table
CREATE TABLE quote_details (
    quote_details_line_item_id INT IDENTITY(1,1) CONSTRAINT PK_quote_details_line_item_id PRIMARY KEY,
    quote_details_quote_id INT,
    quote_details_version INT, -- Added this column to reference both parts of the composite key
    quote_details_product_id INT,
    quote_details_quantity INT,
    quote_details_line_total DECIMAL(10,2),
    CONSTRAINT FK_quote_details_quote_id_version FOREIGN KEY (quote_details_quote_id, quote_details_version) REFERENCES quote(quote_quote_id, quote_version),
    CONSTRAINT FK_quote_details_product_id FOREIGN KEY (quote_details_product_id) REFERENCES products(products_product_id)
);
GO
-- Create Invoice Table
CREATE TABLE invoice (
    invoice_invoice_id INT IDENTITY(1,1) CONSTRAINT PK_invoice_invoice_id PRIMARY KEY,
    invoice_date_generated DATE,
    invoice_date_due DATE,
    invoice_payment_status VARCHAR(50),
    invoice_quote_id INT,
    invoice_quote_version INT, -- Added this column to reference both parts of the composite key
    CONSTRAINT FK_invoice_quote_id_version FOREIGN KEY (invoice_quote_id, invoice_quote_version) REFERENCES quote(quote_quote_id, quote_version)
);
GO
-- Create Invoice Details Table
CREATE TABLE invoice_details (
    invoice_details_line_item_id INT IDENTITY(1,1) CONSTRAINT PK_invoice_details_line_item_id PRIMARY KEY,
    invoice_details_invoice_id INT,
    invoice_details_product_id INT,
    invoice_details_quantity INT,
    invoice_details_line_total DECIMAL(10, 2),
    CONSTRAINT FK_invoice_details_invoice_id FOREIGN KEY (invoice_details_invoice_id) REFERENCES invoice(invoice_invoice_id),
    CONSTRAINT FK_invoice_details_product_id FOREIGN KEY (invoice_details_product_id) REFERENCES products(products_product_id)
);
GO
-- Create Payments Table
CREATE TABLE Payments (
    payments_payment_id INT IDENTITY(1,1) CONSTRAINT PK_payments_payment_id PRIMARY KEY,
    payments_payment_amount DECIMAL(10, 2),
    payments_payment_date DATE,
    payments_payment_method VARCHAR(50),
    payments_invoice_id INT,
    CONSTRAINT FK_payments_invoice_id FOREIGN KEY (payments_invoice_id) REFERENCES invoice(invoice_invoice_id)
);
GO

CREATE PROCEDURE GenerateQuote
    @quote_quote_id INT = NULL,  -- Optional parameter; if NULL, auto-generate
    @quote_customer_id INT,
    @quote_delivery_street VARCHAR(255),
    @quote_delivery_city VARCHAR(50),
    @quote_delivery_state VARCHAR(50),
    @quote_delivery_zip VARCHAR(20),
    @quote_delivery_date DATE,
    @quote_proposal TEXT,
    @quote_expiration DATE,
    @quote_estimate_status VARCHAR(50) = NULL,  -- Optional parameter with default value NULL
    @quote_approved_by_customer VARCHAR(3) = 'No'  -- Optional parameter with default value 'No'
AS
BEGIN
    DECLARE @quote_version INT;

    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Check if the customer exists
        IF NOT EXISTS (SELECT 1 FROM customers WHERE customers_customer_id = @quote_customer_id)
        BEGIN
            THROW 50002, 'Customer ID does not exist.', 1;
        END

        -- Determine the next quote ID if not provided
        IF @quote_quote_id IS NULL
        BEGIN
            SELECT @quote_quote_id = ISNULL(MAX(quote_quote_id), 0) + 1
            FROM quote;
        END

        -- Determine the next version number for the quote
        SELECT @quote_version = ISNULL(MAX(quote_version), 0) + 1
        FROM quote
        WHERE quote_quote_id = @quote_quote_id;

        -- Insert the new quote with the determined version number
        INSERT INTO quote (
            quote_quote_id, 
            quote_version, 
            quote_customer_id, 
            quote_date_updated, 
            quote_delivery_street, 
            quote_delivery_city, 
            quote_delivery_state, 
            quote_delivery_zip, 
            quote_delivery_date, 
            quote_estimate_status, 
            quote_proposal, 
            quote_expiration,
            quote_approved_by_customer  -- New column
        )
        VALUES (
            @quote_quote_id, 
            @quote_version, 
            @quote_customer_id, 
            GETDATE(), 
            @quote_delivery_street, 
            @quote_delivery_city, 
            @quote_delivery_state, 
            @quote_delivery_zip, 
            @quote_delivery_date, 
            ISNULL(@quote_estimate_status, 'No'),  -- Use provided value or default to 'No'
            @quote_proposal, 
            @quote_expiration,
            ISNULL(@quote_approved_by_customer, 'No')  -- Use provided value or default to 'No'
        );

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print the error message
        PRINT 'An error occurred while generating the quote.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO

CREATE PROCEDURE AddItemsToQuoteDetails
    @quote_id INT,
    @product_id INT,
    @quantity INT
AS
BEGIN
    DECLARE @latest_version INT;

    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Determine the latest version of the specified quote
        SELECT @latest_version = MAX(quote_version)
        FROM quote
        WHERE quote_quote_id = @quote_id;

        -- Check if the quote exists
        IF @latest_version IS NULL
        BEGIN
            THROW 50003, 'Quote ID does not exist.', 1;
        END

        -- Check if the product exists
        IF NOT EXISTS (SELECT 1 FROM products WHERE products_product_id = @product_id)
        BEGIN
            THROW 50004, 'Product ID does not exist.', 1;
        END

        -- Insert the item into the quote_details table for the latest version
        INSERT INTO quote_details (
            quote_details_quote_id, 
            quote_details_version, 
            quote_details_product_id, 
            quote_details_quantity
        )
        VALUES (
            @quote_id, 
            @latest_version, 
            @product_id, 
            @quantity
        );

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print the error message
        PRINT 'An error occurred while adding items to quote details.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO


CREATE TRIGGER trg_calculate_quote_details_total
ON quote_details
AFTER INSERT, UPDATE
AS
BEGIN
    -- Update line total after insert or update
    UPDATE qd
    SET qd.quote_details_line_total = qd.quote_details_quantity * p.products_price
    FROM quote_details qd
    JOIN inserted i ON qd.quote_details_line_item_id = i.quote_details_line_item_id
    JOIN products p ON qd.quote_details_product_id = p.products_product_id;
END;
GO

CREATE TRIGGER trg_calculate_invoice_details_total
ON invoice_details
AFTER INSERT, UPDATE
AS
BEGIN
    -- Update line total after insert or update
    UPDATE id
    SET id.invoice_details_line_total = id.invoice_details_quantity * p.products_price
    FROM invoice_details id
    JOIN inserted i ON id.invoice_details_line_item_id = i.invoice_details_line_item_id
    JOIN products p ON id.invoice_details_product_id = p.products_product_id;
END;
GO

CREATE PROCEDURE ProcessPayment
    @payment_amount DECIMAL(10, 2),
    @payment_date DATE,
    @payment_method VARCHAR(50),
    @invoice_id INT
AS
BEGIN
    DECLARE @total_payments DECIMAL(10, 2);
    DECLARE @total_line_items DECIMAL(10, 2);
    DECLARE @sales_tax_rate DECIMAL(5, 2);
    DECLARE @total_tax DECIMAL(10, 2);
    DECLARE @total_due DECIMAL(10, 2);
    DECLARE @remaining_balance DECIMAL(10, 2);

    -- Start the TRY block for error handling
    BEGIN TRY
        -- Start the transaction
        BEGIN TRANSACTION;

        -- Calculate total line items for the invoice
        SELECT @total_line_items = SUM(id.invoice_details_line_total)
        FROM invoice_details id
        WHERE id.invoice_details_invoice_id = @invoice_id;

        -- Get the applicable sales tax rate based on the quote's delivery state
        SELECT @sales_tax_rate = st.sales_tax_tax_rate
        FROM invoice i
        JOIN quote q ON i.invoice_quote_id = q.quote_quote_id AND i.invoice_quote_version = q.quote_version
        JOIN sales_tax st ON q.quote_delivery_state = st.sales_tax_state
        WHERE i.invoice_invoice_id = @invoice_id;

        -- Calculate the total tax
        SET @total_tax = (@total_line_items * @sales_tax_rate) / 100;

        -- Calculate the total due (line items + tax)
        SET @total_due = @total_line_items + @total_tax;

        -- Calculate total payments made for the invoice
        SELECT @total_payments = ISNULL(SUM(p.payments_payment_amount), 0)
        FROM Payments p
        WHERE p.payments_invoice_id = @invoice_id;

        -- Calculate remaining balance due
        SET @remaining_balance = @total_due - @total_payments;

        -- Check if the payment amount matches the full remaining balance (line items + tax)
        IF @payment_amount <> @remaining_balance
        BEGIN
            THROW 50001, 'Payment amount must equal the full remaining balance including taxes.', 1;
        END;

        -- Insert the payment record
        INSERT INTO Payments (payments_payment_amount, payments_payment_date, payments_payment_method, payments_invoice_id)
        VALUES (@payment_amount, @payment_date, @payment_method, @invoice_id);

        -- Recalculate total payments after the new payment
        SELECT @total_payments = SUM(payments_payment_amount)
        FROM Payments
        WHERE payments_invoice_id = @invoice_id;

        -- If total payments equal or exceed total due (line items + taxes), update the payment status to 'Paid'
        IF @total_payments >= @total_due
        BEGIN
            UPDATE invoice
            SET invoice_payment_status = 'Paid'
            WHERE invoice_invoice_id = @invoice_id;
        END;

        -- Commit the transaction
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of an error
        ROLLBACK TRANSACTION;

        -- Print error information
        PRINT 'An error occurred while processing the payment.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO

CREATE PROCEDURE InsertCustomer
    @first_name VARCHAR(50),
    @last_name VARCHAR(50),
    @billing_address VARCHAR(255),
    @billing_city VARCHAR(50),
    @billing_state VARCHAR(50),
    @billing_zip VARCHAR(20),
    @email VARCHAR(100),
    @phone_number VARCHAR(20)
AS
BEGIN
    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Insert the new customer into the customers table
        INSERT INTO customers (
            customers_first_name, 
            customers_last_name, 
            customers_billing_address, 
            customers_billing_city, 
            customers_billing_state, 
            customers_billing_zip, 
            customers_email, 
            customers_phone_number
        )
        VALUES (
            @first_name, 
            @last_name, 
            @billing_address, 
            @billing_city, 
            @billing_state, 
            @billing_zip, 
            @email, 
            @phone_number
        );

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;
        
        -- Return a message indicating success
        PRINT 'Customer inserted successfully.';
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print error information
        PRINT 'An error occurred while inserting the customer.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO

CREATE PROCEDURE InsertOrUpdateDiscount
    @discount_id INT = NULL,  -- Optional parameter; if NULL, a new record is inserted
    @discount_name VARCHAR(100),
    @discount_value DECIMAL(5, 2)
AS
BEGIN
    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Check if the discount already exists
        IF EXISTS (SELECT 1 FROM discounts WHERE discounts_discount_id = @discount_id)
        BEGIN
            -- If discount exists, update it
            UPDATE discounts
            SET discounts_name = @discount_name,
                discounts_discount_value = @discount_value
            WHERE discounts_discount_id = @discount_id;

            PRINT 'Discount updated successfully.';
        END
        ELSE
        BEGIN
            -- If discount does not exist, insert a new record
            INSERT INTO discounts (
                discounts_name, 
                discounts_discount_value
            )
            VALUES (
                @discount_name, 
                @discount_value
            );

            PRINT 'Discount inserted successfully.';
        END

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print error information
        PRINT 'An error occurred while inserting or updating the discount.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO

CREATE PROCEDURE InsertOrUpdateProduct
    @product_id INT = NULL,  -- Optional parameter; if NULL, a new product will be inserted
    @product_name VARCHAR(100),
    @product_description TEXT,
    @product_price DECIMAL(10, 2),
    @product_unit_of_measure VARCHAR(50)
AS
BEGIN
    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Check if the product already exists
        IF EXISTS (SELECT 1 FROM products WHERE products_product_id = @product_id)
        BEGIN
            -- If the product exists, update it
            UPDATE products
            SET products_name = @product_name,
                products_description = @product_description,
                products_price = @product_price,
                products_unit_of_measure = @product_unit_of_measure
            WHERE products_product_id = @product_id;

            PRINT 'Product updated successfully.';
        END
        ELSE
        BEGIN
            -- If the product does not exist, insert a new record
            INSERT INTO products (
                products_name, 
                products_description, 
                products_price, 
                products_unit_of_measure
            )
            VALUES (
                @product_name, 
                @product_description, 
                @product_price, 
                @product_unit_of_measure
            );

            PRINT 'Product inserted successfully.';
        END

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print error information
        PRINT 'An error occurred while inserting or updating the product.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO

CREATE PROCEDURE GenerateInvoiceFromQuote
    @quote_id INT
AS
BEGIN
    DECLARE @quote_version INT;
    DECLARE @quote_approved VARCHAR(3);
    DECLARE @invoice_id INT;

    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Check if the quote exists and is approved by the customer
        SELECT @quote_version = MAX(quote_version),
               @quote_approved = MAX(quote_approved_by_customer)
        FROM quote
        WHERE quote_quote_id = @quote_id;

        IF @quote_version IS NULL
        BEGIN
            THROW 50005, 'Quote does not exist.', 1;
        END

        IF @quote_approved <> 'Yes'
        BEGIN
            THROW 50006, 'Quote must be approved by the customer before generating an invoice.', 1;
        END

        -- Insert a new invoice record
        INSERT INTO invoice (
            invoice_date_generated, 
            invoice_date_due, 
            invoice_payment_status, 
            invoice_quote_id, 
            invoice_quote_version
        )
        VALUES (
            GETDATE(), 
            DATEADD(DAY, 30, GETDATE()),  -- Assumes a default due date 30 days from now
            'Unpaid', 
            @quote_id, 
            @quote_version
        );

        -- Get the new invoice ID
        SELECT @invoice_id = SCOPE_IDENTITY();

        -- Insert corresponding invoice details based on the quote details
        INSERT INTO invoice_details (
            invoice_details_invoice_id, 
            invoice_details_product_id, 
            invoice_details_quantity
        )
        SELECT 
            @invoice_id, 
            quote_details_product_id, 
            quote_details_quantity
        FROM quote_details
        WHERE quote_details_quote_id = @quote_id
          AND quote_details_version = @quote_version;

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;

        PRINT 'Invoice generated successfully.';
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print error information
        PRINT 'An error occurred while generating the invoice from the quote.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO

CREATE PROCEDURE AddOrUpdateInvoiceDetails
    @invoice_id INT,
    @product_id INT,
    @quantity INT
AS
BEGIN
    BEGIN TRY
        -- Start a transaction to ensure atomicity
        BEGIN TRANSACTION;

        -- Check if the invoice detail already exists
        IF EXISTS (SELECT 1 FROM invoice_details 
                   WHERE invoice_details_invoice_id = @invoice_id 
                     AND invoice_details_product_id = @product_id)
        BEGIN
            -- If the invoice detail exists, update it
            UPDATE invoice_details
            SET invoice_details_quantity = @quantity
            WHERE invoice_details_invoice_id = @invoice_id 
              AND invoice_details_product_id = @product_id;

            PRINT 'Invoice details updated successfully.';
        END
        ELSE
        BEGIN
            -- If the invoice detail does not exist, insert a new record
            INSERT INTO invoice_details (
                invoice_details_invoice_id, 
                invoice_details_product_id, 
                invoice_details_quantity
            )
            VALUES (
                @invoice_id, 
                @product_id, 
                @quantity
            );

            PRINT 'Item added to invoice details successfully.';
        END

        -- Commit the transaction if everything is successful
        COMMIT TRANSACTION;
    END TRY
    BEGIN CATCH
        -- Rollback the transaction in case of any error
        ROLLBACK TRANSACTION;

        -- Print error information
        PRINT 'An error occurred while managing invoice details.';
        PRINT ERROR_MESSAGE();
    END CATCH;
END;
GO


-- Insert sample data into sales_tax table
INSERT INTO sales_tax (sales_tax_state, sales_tax_tax_rate)
VALUES 
('TX', 3.00),
('CA', 7.25);

-- Insert sample customers using InsertCustomer procedure
EXEC InsertCustomer 
    @first_name = 'John', 
    @last_name = 'Doe', 
    @billing_address = '123 Elm Street', 
    @billing_city = 'Austin', 
    @billing_state = 'TX', 
    @billing_zip = '78701', 
    @email = 'john.doe@example.com', 
    @phone_number = '555-1234';

EXEC InsertCustomer 
    @first_name = 'Jane', 
    @last_name = 'Smith', 
    @billing_address = '456 Oak Avenue', 
    @billing_city = 'Austin', 
    @billing_state = 'TX', 
    @billing_zip = '78702', 
    @email = 'jane.smith@example.com', 
    @phone_number = '555-5678';

GO

-- Insert or update discounts using InsertOrUpdateDiscount procedure
EXEC InsertOrUpdateDiscount 
    @discount_id = NULL,  -- NULL to insert a new discount
    @discount_name = 'Holiday Discount', 
    @discount_value = 10.00;

EXEC InsertOrUpdateDiscount 
    @discount_id = NULL, 
    @discount_name = 'Loyalty Discount', 
    @discount_value = 5.00;

go

-- Insert or update products using InsertOrUpdateProduct procedure
EXEC InsertOrUpdateProduct 
    @product_id = NULL,  -- NULL to insert a new product
    @product_name = 'Lawn Mowing', 
    @product_description = 'Standard lawn mowing service', 
    @product_price = 50.00, 
    @product_unit_of_measure = 'Acre';

EXEC InsertOrUpdateProduct 
    @product_id = Null, 
    @product_name = 'Tree Trimming', 
    @product_description = 'Tree trimming and pruning service', 
    @product_price = 100.00, 
    @product_unit_of_measure = 'Tree';

EXEC InsertOrUpdateProduct 
    @product_id = Null, 
    @product_name = 'Mowing', 
    @product_description = 'Standard mow', 
    @product_price = 100.00, 
    @product_unit_of_measure = 'Acre';

GO

-- Generate a quote using GenerateQuote procedure
EXEC GenerateQuote 
    @quote_quote_id = NULL,  -- NULL to auto-generate a new quote ID
    @quote_customer_id = 1,  
    @quote_delivery_street = '789 Pine Street',
    @quote_delivery_city = 'Austin',
    @quote_delivery_state = 'TX',
    @quote_delivery_zip = '78703',
    @quote_delivery_date = '2024-09-10',
    @quote_proposal = 'Proposal details here...',
    @quote_expiration = '2024-09-30',
    @quote_estimate_status = 'Pending',  
    @quote_approved_by_customer = 'Yes';  


GO

-- Add items to quote details using AddItemsToQuoteDetails procedure
EXEC AddItemsToQuoteDetails 
    @quote_id = 1,  
    @product_id = 1,  
    @quantity = 4; 

EXEC AddItemsToQuoteDetails 
    @quote_id = 1, 
    @product_id = 2, 
    @quantity = 3;

-- Generate a second quote
EXEC GenerateQuote 
    @quote_quote_id = NULL,  -- NULL to auto-generate a new quote ID
    @quote_customer_id = 2,  
    @quote_delivery_street = '456 Oak Avenue',
    @quote_delivery_city = 'Austin',
    @quote_delivery_state = 'TX',
    @quote_delivery_zip = '78703',
    @quote_delivery_date = '2024-09-10',
    @quote_proposal = 'Proposal details here...',
    @quote_expiration = '2024-09-30',
    @quote_estimate_status = 'Pending',  
    @quote_approved_by_customer = 'No';  


GO

-- Add items to quote details using AddItemsToQuoteDetails procedure
EXEC AddItemsToQuoteDetails 
    @quote_id = 2,  
    @product_id = 1,  
    @quantity = 6; 

EXEC AddItemsToQuoteDetails 
    @quote_id = 2, 
    @product_id = 2, 
    @quantity = 5;

-- Generate an invoice from a quote using GenerateInvoiceFromQuote procedure
EXEC GenerateInvoiceFromQuote 
    @quote_id = 1;  

GO

-- Add or update items in invoice details using AddOrUpdateInvoiceDetails procedure
EXEC AddOrUpdateInvoiceDetails 
    @invoice_id = 1,
    @product_id = 1,
    @quantity = 4;  -- Example quantity that's getting updated


EXEC AddOrUpdateInvoiceDetails 
    @invoice_id = 1, 
    @product_id = 2, 
    @quantity = 5;

GO

-- View Invoices with unpaid Balances
SELECT 
    i.invoice_invoice_id,
    c.customers_first_name + ' ' + c.customers_last_name AS customer_name,
    SUM(id.invoice_details_line_total) AS line_item_total,
    st.sales_tax_tax_rate AS sales_tax_rate,
    (SUM(id.invoice_details_line_total) * (st.sales_tax_tax_rate / 100)) AS sales_tax,
    (SUM(id.invoice_details_line_total) + (SUM(id.invoice_details_line_total) * (st.sales_tax_tax_rate / 100))) AS total_amount_due_with_tax
FROM 
    invoice i
JOIN 
    quote q ON i.invoice_quote_id = q.quote_quote_id AND i.invoice_quote_version = q.quote_version
JOIN 
    customers c ON q.quote_customer_id = c.customers_customer_id
JOIN 
    sales_tax st ON q.quote_delivery_state = st.sales_tax_state
JOIN 
    invoice_details id ON i.invoice_invoice_id = id.invoice_details_invoice_id
WHERE 
    i.invoice_payment_status = 'Unpaid'
GROUP BY 
    i.invoice_invoice_id,
    c.customers_first_name,
    c.customers_last_name,
    st.sales_tax_tax_rate;

-- Process a payment using ProcessPayment procedure
EXEC ProcessPayment 
    @payment_amount = 721.00,  -- Example payment amount
    @payment_date = '2024-09-01',  -- Example payment date
    @payment_method = 'Credit Card',  -- Example payment method
    @invoice_id = 1;  


-- Query to see unapproved quotes with calculated totals
WITH LatestQuote AS (
    SELECT 
        quote_quote_id,
        MAX(quote_version) AS latest_version
    FROM 
        quote
    GROUP BY 
        quote_quote_id
)
SELECT 
    q.quote_quote_id,
    c.customers_first_name + ' ' + c.customers_last_name AS customer_name,
    q.quote_date_updated,
    SUM(qd.quote_details_quantity * p.products_price) AS line_item_total,
    st.sales_tax_tax_rate AS sales_tax_rate,
    (SUM(qd.quote_details_quantity * p.products_price) * (st.sales_tax_tax_rate / 100)) AS sales_tax,
    (SUM(qd.quote_details_quantity * p.products_price) + (SUM(qd.quote_details_quantity * p.products_price) * (st.sales_tax_tax_rate / 100))) AS total_quote_amount
FROM 
    quote q
JOIN 
    LatestQuote lq ON q.quote_quote_id = lq.quote_quote_id AND q.quote_version = lq.latest_version
JOIN 
    customers c ON q.quote_customer_id = c.customers_customer_id
JOIN 
    sales_tax st ON c.customers_billing_state = st.sales_tax_state
JOIN 
    quote_details qd ON q.quote_quote_id = qd.quote_details_quote_id AND q.quote_version = qd.quote_details_version
JOIN 
    products p ON qd.quote_details_product_id = p.products_product_id
WHERE 
    q.quote_approved_by_customer = 'No'  -- Unapproved quotes
GROUP BY 
    q.quote_quote_id, 
    c.customers_first_name, 
    c.customers_last_name, 
    q.quote_date_updated, 
    st.sales_tax_tax_rate;


