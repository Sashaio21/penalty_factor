import axios from "axios";

const intance = axios.create({
    baseURL: 'https://solution-penalty-server-1.onrender.com'
    // baseURL: 'http://localhost:4444'
});
 

intance.interceptors.request.use((config)=>{
    config.headers.authorization = window.localStorage.getItem('token');
    return config
});

export default intance;