import { configureStore } from "@reduxjs/toolkit";
import { counterReducer } from "./slices/counterSlices";
import { useReducer } from "react";

const store = configureStore({
    reducer:{
        counter: counterReducer,
        user: useReducer
    }
})

export default store;