// import { createSlice } from "@reduxjs/toolkit";

// export const fetchUser = createAsyncThunk('user/fetchUser', async (userId) => {
//     const response = await fetch(`https://jsonplaceholder.typicode.com/users/${userId}`);
//     return response.json();
//   });


// const userSlice = createSlice({
//     name: "user",
//     initialState:{
//         user: null,
//         status: 'idle',
//         error: null
//     },
//     reducers:{},
//     extraReducers: {
//         [fetchUser.pending]: ()=>{
//             state.status = 'loading'
//             state.user = null
//         },
//         [fetchUser.fulfilled]: (state)=>{
//             state.status = 'succeeded'
//             state.user = action.payload;
//         }
//     }
// })

// export const userReducer = userSlice.reducer;