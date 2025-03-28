import '../App.css';
import './Header.css'
import { Link } from "react-router-dom"
import { Button } from '@mui/material';
import { useEffect, useState } from 'react';
import { Navigate } from 'react-router-dom';
import axios from '../axios';
import Typography from '@mui/material';

function Header(){
    const [change, setChange] = useState(false)
    const [token, setToken] = useState(false)
    const [user, setUser] = useState(false)

    useEffect(()=>{
        console.log("Header")
        setToken(window.localStorage.getItem('token'))
        if (window.localStorage.getItem('token')) {
            axios.get('/auth/me')
            .then((data)=>{
                setChange(true)
                setUser(data.data)
                console.log(data.data)
            }) 
        }
    },[window.localStorage.getItem('token')])
    

    function exitAccaunt(){
        // 
        if (window.confirm("Вы хотите выйти?")) {
            window.localStorage.removeItem('token');
            setChange(false)
            setUser(false)
        }
    }
    
    return (
        <div className="App-header ">
            {!change ? 
            <div className='header'>
                <Link to={'/'}>На главную</Link>
                <div className='userIU'>
                    <Link to={'/login'}><Button>Вход</Button></Link>
                    <Link to={'/register'}><Button>Регистрация</Button></Link>
                </div>
            </div> :  
            <div className='header'>
                <Link to={'/'}>На главную</Link>
                <div className='userIU'>
                    <p style={{fontSize: "18px", alignSelf:"flex-start"}}>{user.firstName} {user.name}</p>
                    <Link to={'/login'} style={{marginTop:"12px", marginLeft:"20px"}} onClick={()=>exitAccaunt()}><Button>Выйти</Button></Link>
                </div>
            </div>
            }
            
        </div>
    )
}
export default Header