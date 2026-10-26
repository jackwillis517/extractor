from typing import Literal

from pydantic import BaseModel


class JobPosting(BaseModel):
    title: str
    company: str
    location: Literal["remote", "hybrid", "onsite"]
    is_remote: bool
    description: str
    min_salary: int
    max_salary: int
    requirements: str


class JobSearchResults(BaseModel):
    jobs: list[JobPosting]


class LineItem(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(BaseModel):
    invoice_number: str
    vendor: str
    customer: str
    invoice_date: str
    due_date: str
    currency: str
    subtotal: float
    discount: float
    tax: float
    total: float
    line_items: list[LineItem]
