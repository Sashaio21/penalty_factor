import '../App.css';
import { useForm } from 'react-hook-form';
import { useState } from 'react';
import { Input, Button } from '@mui/material';
import { Link, Navigate } from 'react-router-dom';
import axios from '../axios'
import {CircularProgress} from '@mui/material';


function Login() {
  const {register, handleSubmit, formState:{errors}} = useForm();
  const [token, setToken] = useState()
  const [errorLogin, setErrorLogin] = useState(false)
  const [visibleLoading, setVisibleLoading] = useState(false)


  function LoginUser(data){
    setVisibleLoading(true)
    const dataAuth =  axios.post('/auth/login', data)
      .then((res)=>{
        if (res.data.token) {
          setErrorLogin(false)
          window.localStorage.setItem('token', res.data.token)
          setToken(res.data.token)
        }
      })
      .catch((err)=>{
        setErrorLogin(err.response.data.message)
        console.log(err.response.data.message)
      }).finally(()=>{
        setVisibleLoading(false)
      })
    
  }

  if (window.localStorage.getItem('token')) {
    return <Navigate to={'/'}/> 
  }


  return (
    <div style={{display: "flex", justifyContent: "center"}}>
      <div style={{display: "flex", flexDirection: "column", justifyContent:"center"}}>
      <p style={{alignSelf:"center"}}>Авторизация</p>
        <form onSubmit={handleSubmit(LoginUser)}>
            <div style={{display: "flex", flexDirection: "column", width: "250px"}}>
              <Input
                placeholder='email'
                name='email'
                {...register('email')}
              />
              <Input
                placeholder='пароль'
                name='password'
                type='password'
                {...register('password')}
              />
              {!visibleLoading ? (
                <Button type='submit'>Войти</Button>
              ):(
                <div style={{display:"flex", marginLeft:"92px"}}>
                  <Button type='submit'>Войти</Button>
                  <CircularProgress size={20} style={{marginTop:"10px"}}></CircularProgress>
                </div>
              )}
            </div>
        </form>
        {errorLogin?(<p style={{alignSelf:"center", color:"red"}}>{errorLogin}</p>):<></>}
        <p style={{alignSelf:"center"}}>Нет аккаунт?</p>
        <Link to={'/register'} style={{alignSelf:"center"}}><Button>Зарегистрироваться</Button></Link>
      </div>
    </div>
  );
}

export default Login;