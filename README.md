# Student Performance Analysis System
### Using Python, NumPy, and Pandas

**Course:** Artificial Intelligence, Machine Learning & Deep Learning  
**Student Name:** Muhammad Umar  
**Student ID:** 09  
**Role:** Junior AI Engineer (Trainee)  

---

## 📌 Project Overview
This project focuses on Exploratory Data Analysis (EDA) of a student performance dataset containing academic records for **500 students** across **11 distinct features**. Working in the role of a Junior AI Engineer, the objective is to inspect, clean, filter, and analyze underlying academic patterns using **Python**, **Pandas**, and **NumPy** before developing predictive machine learning models.

---

## 📂 Dataset Description (`students_performance_dataset_500.csv`)
The dataset consists of 500 rows and 11 columns:
1. **Student_ID**: Integer — Unique identifier assigned to each student.
2. **Name**: Text — Full name of the student.
3. **Age**: Integer — Age of the student in years.
4. **Gender**: Text — Male / Female.
5. **Department**: Text — Academic department (e.g., CS, AI, SE).
6. **Semester**: Integer — Current semester.
7. **Attendance**: Numeric (%) — Overall class attendance percentage.
8. **Assignment_Marks**: Numeric — Marks obtained in assignments.
9. **Quiz_Marks**: Numeric — Marks obtained in quizzes.
10. **Mid_Marks**: Numeric — Marks obtained in the mid-term examination.
11. **Final_Marks**: Numeric — Marks obtained in the final examination.

---

## 🚀 Implementation Steps & Code Structure

The analysis is broken down into modular steps covering:
- **Part 1: Loading the Dataset** — Importing CSV data via Pandas and inspecting dimensions, headers, head, and tail rows.
- **Part 2: Exploring the Dataset** — Calculating total student counts, unique departments, and gender demographics.
- **Part 3: Column Selection** — Extracting specific feature subsets (`Name`, `Department`, `Attendance`, `Final_Marks`).
- **Part 4: Row Selection** — Positional (`iloc`) and label-based (`loc`) index slicing.
- **Part 5: Conditional Selection** — Filtering rows based on logical criteria (e.g., Final Marks > 40, Attendance > 90%, Department filters).
- **Part 6: NumPy Operations** — Converting data into NumPy arrays to compute statistical metrics ($\mu, \min, \max$) and perform element-wise transformations ($\sqrt{x}, 2x$).
- **Part 7: Final Student Report** — Generating the mandatory summary report.

---

## 📊 Final Program Output Summary

```text
==================================================
STUDENT PERFORMANCE REPORT
==================================================
Total Students: 500
Total Departments: 5
Average Attendance: 77.07%
Average Final Marks: 35.08
Highest Final Marks: 50
Lowest Final Marks: 20
==================================================
