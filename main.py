from fastapi import FastAPI
app = FastAPI()

mock_db = {
    "123": "Shipped — arriving Friday via FedEx (#TRK789).",
    "456": "Processing — dispatches in 2 business days.",
    "ORD-001": "Cancelled — refund issued within 5–7 business days.",
}

@app.get("/orders/{order_id}")
def get_order(order_id: str):
    return {"order_id": order_id, "status": mock_db.get(order_id, "Order not found.")}