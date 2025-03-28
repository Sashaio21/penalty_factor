import '../App.css';
import React from 'react';
import { Card, CardContent, Avatar, Typography, Button, Box } from '@mui/material';
import { Link } from 'react-router-dom';

function Profile() {
  return (
    <Card sx={{ maxWidth: 345, margin: 'auto', mt: 5, padding: 3 }}>
      <Box display="flex" justifyContent="center">
        <Avatar
          alt="John Doe"
          src="https://via.placeholder.com/150"
          sx={{ width: 100, height: 100 }}
        />
      </Box>

      <CardContent>
        <Typography variant="h5" align="center" gutterBottom>
          John Doe
        </Typography>

        <Typography variant="body2" color="textSecondary" align="center">
          Web Developer | Enthusiast | Lifelong Learner
        </Typography>

        <Box display="flex" justifyContent="center" alignItems="center" mt={2}>
          <Typography variant="body2">johndoe@example.com</Typography>
        </Box>

        <Box display="flex" justifyContent="center" mt={3}>
          <Link to={'/support'}>
            <Button variant="outlined" color="secondary">
              Тех.поддержка
            </Button>
          </Link>
        </Box>
      </CardContent>
    </Card>
  );
}

export default Profile;