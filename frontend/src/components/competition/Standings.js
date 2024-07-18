import React, { useEffect, useState } from "react";
import { TableContainer, Table, TableHead, TableRow, TableCell, TableBody } from '@mui/material';
import Paper from '@mui/material/Paper';

const Standings = ({ initialBalance, totalAssets, aiTotalAssets }) => {
    const [standings, setStandings] = useState([]);

    useEffect(() => {
        const newStandings = [
            { "Competitor": "Player", "Income": totalAssets - initialBalance, "Rank": totalAssets >= aiTotalAssets ? 1 : 2 },
            { "Competitor": "AI", "Income": aiTotalAssets - initialBalance, "Rank": aiTotalAssets >= totalAssets ? 1 : 2 }
        ].sort((a, b) => a.Rank - b.Rank);

        setStandings(newStandings);
    }, [initialBalance, totalAssets, aiTotalAssets]);

    return (
        <TableContainer component={Paper} style={{backgroundColor:"rgba(180, 180, 180, 0.8)"}}>
            <Table sx={{ minWidth: 650 }} aria-label="simple table">
                <TableHead>
                    <TableRow>
                        <TableCell align="center">Competitor</TableCell>
                        <TableCell align="center">Rank</TableCell>
                        <TableCell align="center">Income</TableCell>
                    </TableRow>
                </TableHead>
                <TableBody>
                    {standings.map((row) => (
                        <TableRow
                            key={row.Competitor}
                            sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                        >
                            <TableCell align="center">
                                {row.Competitor}
                            </TableCell>
                            <TableCell align="center">{row.Rank}</TableCell>
                            <TableCell align="center">{row.Income.toFixed(2)}</TableCell>
                        </TableRow>
                    ))}
                </TableBody>
            </Table>
        </TableContainer>
    )
}

export default Standings;
