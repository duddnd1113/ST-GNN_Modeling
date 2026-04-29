# 🌫️ Dynamic PM10 Mapping & Route Optimization using ST-GNN + LUR

## Engineering Industry Competition Project

This project focuses on solving real-world urban air pollution problems through **AI-based modeling and optimization**.

---

## Overview

We build a **dynamic PM10 mapping system** and use it to solve two key optimization problems:

### 1️ Road Cleaning Vehicle Routing Optimization

→ Minimize overall city pollution by optimizing suction vehicle routes

### 2️ Low-Exposure Path Optimization

→ Provide routes for pedestrians that minimize pollution exposure

---

## Motivation

Urban air pollution is not just a prediction problem.

It is a **decision-making problem**:

* Where should cleaning vehicles operate?
* How should people move to reduce exposure?

To answer these questions, we need:
→ **Dynamic, high-resolution pollution maps**

---

## Method

### 1. ST-GNN (Transport Modeling)

* Models how pollution spreads across locations
* Uses wind-aware directed graph
* Captures temporal dynamics via GRU

---

### 2. LUR (Local Modeling)

* Captures local pollution generation factors
* Includes:

  * Road density
  * Land use
  * Population
  * Urban structure

---

### 3. Hybrid Model

PM = f(Geographic Features, ST-GNN Representation)

→ Combines **transport dynamics + local generation**

---

### 4. Dynamic PM Map

We generate a **city-wide pollution map** that changes over time.

---

## Optimization Tasks

### 🚗 (1) Suction Vehicle Route Optimization

Goal:

* Reduce overall city PM levels

Approach:

* Identify high-pollution areas
* Optimize vehicle routes to maximize cleaning efficiency

---

### 🚶 (2) Low-Exposure Path Optimization

Goal:

* Minimize human exposure to PM

Approach:

* Use dynamic PM map
* Find paths with lowest cumulative pollution

---

## Key Features

* Edge-aware ST-GNN with physical interpretation
* Dynamic pollution map generation
* Two real-world optimization applications:

  * Vehicle routing
  * Human path optimization

---

## 📊 Data

* Air quality monitoring stations (Seoul)
* Meteorological data (wind, temperature)
* Geographic features (road density, land use)

---

## Key Insight

This project reframes air pollution modeling as:

> **A decision-support system for urban optimization, not just prediction**

---

## 👤 Author

Yonsei University
Industrial Engineering & Urban Planning and Engineering & Quantitative Risk Management

---
